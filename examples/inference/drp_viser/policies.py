"""The checkpoints of ~/drp/ckpt_active served by one vlagents `drp` server each, used from the panel process.

    pool = PolicyPool.from_dir(log_dir=...)      # ports 20101.. (8080 and 20000 are taken on this station)
    pool.start_background(first="<name>")        # starts/adopts the servers, waits for them in parallel
    chunk = chunk_of(pool[name].act(make_obs(q, goal, cams, K, step=k)))   # (chunk_size, 7) absolute joints [rad]
    pool.stop_all()                              # SIGINT -> clean AgentService shutdown

Every server answers name() == "drp", so a server's checkpoint is verified through the agent's meta query
(vlagents fork, branch drp-agent: info["drp_query"] = "meta"). Servers run in their own session, so a Ctrl-C in the
panel's terminal does not reach them; the panel stops them itself (also on a repeated Ctrl-C), and a later panel
stops servers an earlier one left running (killed panel) before starting its own.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import json_numpy
import numpy as np
import rpyc
from rcs.camera.interface import BaseCameraSet
from vlagents.client import RemoteAgent, dataclass_from_dict
from vlagents.policies.interface import Act, Obs, SingleObs

logger = logging.getLogger(__name__)

POLICY_PYTHON = Path.home() / "drp/envs/drp_vlagents/bin/python"
CKPT_ROOT = Path.home() / "drp/ckpt_active"
BASE_PORT = 20101
ROBOT_KEY = "right"
#: RCS depth is uint16 at 1/DEPTH_SCALE m. float32(0.001) is also the D4xx depth unit, so multiplying by it gives the
#: same float32 metres as z16 * get_depth_scale() bit for bit (dividing by 1000 differs in the last bit, which moves
#: odd-millimetre depths one 2 mm policy step down).
MM_TO_M = np.float32(1.0 / BaseCameraSet.DEPTH_SCALE)


# ------------------------------------------------------------------------------------------------- translation
def rcs_frames_to_cameras(frames: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """obs["frames"] of the RCS CameraSetWrapper -> (cameras, intrinsics) in the drp contract: "<slot>" RGB uint8
    HxWx3, "<slot>_depth" float32 metres on the colour grid (0 = no return), K 3x3 of the colour stream."""
    cams: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        cams[name] = np.ascontiguousarray(frame["rgb"]["data"])
        if frame["rgb"].get("intrinsics") is not None:
            intrinsics[name] = np.asarray(frame["rgb"]["intrinsics"], dtype=np.float32)[:3, :3]
        if "depth" in frame:
            depth = np.asarray(frame["depth"]["data"])
            if depth.ndim == 3:
                depth = depth[..., 0]
            cams[f"{name}_depth"] = depth.astype(np.float32) * MM_TO_M
    return cams, intrinsics


POLICY_W, POLICY_H = 320, 240


def to_policy_size(cams: dict, intrinsics: dict, crop: float = 1.0) -> tuple[dict, dict]:
    """Centre-crops (fraction of width and height; 1.0 = whole frame) and resizes every slot's RGB-D to the policy's
    320x240, carrying K through the same crop and scale (pixel centres at integer coordinates). RGB is area-averaged,
    depth sampled nearest: interpolating across an edge would invent surfaces between foreground and background.
    Frames already 320x240 with crop 1.0 pass through unchanged (the D415 rig)."""
    from PIL import Image

    out_c, out_K, boxes = {}, {}, {}
    for key, img in cams.items():
        h, w = img.shape[:2]
        if (w, h) == (POLICY_W, POLICY_H) and crop >= 1.0:
            out_c[key] = img
            continue
        ch, cw = int(round(h * min(crop, 1.0))), int(round(w * min(crop, 1.0)))
        top, left = (h - ch) // 2, (w - cw) // 2
        boxes[key] = (top, left, ch, cw)
        img = img[top:top + ch, left:left + cw]
        if key.endswith("_depth"):
            ys = np.minimum(((np.arange(POLICY_H) + 0.5) * ch / POLICY_H).astype(int), ch - 1)
            xs = np.minimum(((np.arange(POLICY_W) + 0.5) * cw / POLICY_W).astype(int), cw - 1)
            out_c[key] = np.ascontiguousarray(img[ys][:, xs])
        elif (ch, cw) == (2 * POLICY_H, 2 * POLICY_W):  # the D455 at 640x480: exact 2x2 box average
            out_c[key] = (img.reshape(POLICY_H, 2, POLICY_W, 2, 3).mean(axis=(1, 3)) + 0.5).astype(np.uint8)
        else:
            out_c[key] = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((POLICY_W, POLICY_H), Image.BOX))
    for slot, K in intrinsics.items():
        if slot not in boxes:
            out_K[slot] = K
            continue
        top, left, ch, cw = boxes[slot]
        sx, sy = POLICY_W / cw, POLICY_H / ch
        out_K[slot] = np.array([[K[0, 0] * sx, 0.0, (K[0, 2] - left + 0.5) * sx - 0.5],
                                [0.0, K[1, 1] * sy, (K[1, 2] - top + 0.5) * sy - 0.5],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
    return out_c, out_K


def make_obs(joints, goal_joints, cameras: dict, intrinsics: dict | None, *, step: int = 0, episode: int = 0,
             camera_keys: list[str] | None = None, **extra_info) -> Obs:
    """A fresh Obs for one act() call (RemoteAgent.act rewrites the Obs it is given)."""
    info = {"step": int(step), "episode": int(episode), "intrinsics": dict(intrinsics or {}), **extra_info}
    if goal_joints is not None:
        info["goal_joints"] = np.asarray(goal_joints, dtype=np.float64).reshape(7)
    cams = {k: v for k, v in cameras.items() if camera_keys is None or k in camera_keys}
    return Obs(obs={ROBOT_KEY: SingleObs(cameras=cams, joints=np.asarray(joints, dtype=np.float32), info=info)})


def chunk_of(act: Act) -> np.ndarray:
    """(chunk_size, 7) float64 absolute joint targets; raises on an empty / non-finite / query reply."""
    if not act.acts:
        raise ValueError("empty action chunk")
    chunk = np.stack([np.asarray(step[ROBOT_KEY].action, dtype=np.float64) for step in act.acts])
    if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.all(np.isfinite(chunk)):
        raise ValueError(f"not an executable chunk: shape {chunk.shape}, finite {np.all(np.isfinite(chunk))}")
    return chunk


def first_info(act: Act) -> dict:
    return dict(act.acts[0][ROBOT_KEY].info or {}) if act.acts else {}


def read_ckpt_info(root: Path) -> dict[str, dict]:
    """ckpt_info.json next to the checkpoints: {name: {index, short, role, hard, simple, ood3x, ood5x, collision}}."""
    path = Path(root) / "ckpt_info.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        logger.exception("could not parse %s", path)
        return {}


def _nominal_K(width: int = 320, height: int = 240) -> np.ndarray:
    fx, fy = width / 2 / np.tan(np.deg2rad(54.8) / 2), height / 2 / np.tan(np.deg2rad(42.5) / 2)
    return np.array([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)


def _warmup_obs(meta: dict) -> Obs:
    """A valid observation for every input the checkpoint reads (grey RGB, a 1 m wall, nominal K)."""
    h, w = meta.get("policy_hw", [240, 320])
    q = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, 0.0])
    cams = {s: np.full((h, w, 3), 128, np.uint8) for s in meta["rgb_slots"]}
    cams.update({f"{s}_depth": np.ones((h, w), np.float32) for s in meta["depth_slots"]})
    return make_obs(q, q, cams, {s: _nominal_K(w, h) for s in meta.get("intrinsics_slots", [])}, step=1)


def _listener(port: int):
    """(pid, command line) of the process listening on a local TCP port, or None."""
    out = subprocess.run(["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    for pid in re.findall(r"pid=(\d+)", out):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        return int(pid), cmd
    return None


def _other_panels() -> list:
    """PIDs of other running DRP panels."""
    out = subprocess.run(["pgrep", "-f", r"python[0-9.]* .*drp_viser_app\.py"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split() if int(p) != os.getpid()]


def _stop_pid(pid: int, timeout: float = 5.0) -> None:
    """SIGINT (clean rpyc shutdown), then SIGTERM, then SIGKILL."""
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not Path(f"/proc/{pid}").exists():
                return
            time.sleep(0.1)


# ------------------------------------------------------------------------------------------------------ client
class GuiRemoteAgent(RemoteAgent):
    """RemoteAgent with a bounded request time and without the silent reconnect + resend inside act()."""

    def __init__(self, *args, request_timeout: float = 2.0, **kwargs):
        self.request_timeout = float(request_timeout)
        super().__init__(*args, **kwargs)

    def _connect(self):
        self.c = rpyc.connect(self.host, self.port, config={
            "allow_pickle": True, "allow_public_attrs": True, "sync_request_timeout": self.request_timeout})
        assert self.model == self.c.root.name()

    def act(self, obs: Obs) -> Act:
        payload = json_numpy.dumps(asdict(self._process(obs)))
        return dataclass_from_dict(Act, json_numpy.loads(self.c.root.act(payload)))

    def forget_shared_memory(self):
        """Close and unlink the shm segments, tolerating names a killed server's resource tracker already unlinked."""
        for shm in self._shm.values():
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, BufferError):
                pass
        self._shm = {}

    def close(self):
        self.forget_shared_memory()
        if self.c is not None:
            try:
                self.c.close()
            except Exception:  # noqa: BLE001
                pass


# -------------------------------------------------------------------------------------------------- one server
class PolicyServer:
    def __init__(self, ckpt_dir: Path, port: int, log_dir: Path, request_timeout: float = 2.0):
        self.ckpt_dir = Path(ckpt_dir)
        self.name = self.ckpt_dir.name
        self.port = int(port)
        self.log_path = Path(log_dir) / f"drp_{self.port}_{self.name}.log"
        self.request_timeout = request_timeout
        self.proc: subprocess.Popen | None = None
        self.agent: GuiRemoteAgent | None = None
        self.meta: dict | None = None
        self.error: str | None = None
        self.lock = threading.Lock()  # one request in flight per server (shm buffers + rpyc connection)

    @property
    def ready(self) -> bool:
        return self.agent is not None and self.meta is not None and self.error is None

    def _port_open(self) -> bool:
        with socket.socket() as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def start(self):
        """Starts this checkpoint's server. A port still in use (see PolicyPool.clear_leftovers) fails this server
        with the reason instead of reusing whatever serves there."""
        if self._port_open():
            owner = _listener(self.port)
            self.error = (f"port {self.port} is in use"
                          + (f" by pid {owner[0]} ({owner[1][:80]})" if owner else "")
                          + ": another panel's policy server? stop that panel first")
            logger.error("policy server %s not started: %s", self.name, self.error)
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(POLICY_PYTHON), "-m", "vlagents", "start-server", "drp", "--host", "localhost", "--port",
               str(self.port), "--kwargs", json.dumps({"checkpoint_path": str(self.ckpt_dir)})]
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _log_tail(self, n: int = 15) -> str:
        try:
            return "\n".join(self.log_path.read_text().splitlines()[-n:])
        except OSError:
            return "(no log)"

    def wait_ready(self, timeout: float = 180.0, warmup: bool = True) -> dict:
        if self.error:  # not started (port in use)
            raise RuntimeError(self.error)
        deadline = time.monotonic() + timeout
        while self.agent is None:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"server exited ({self.proc.returncode}), log {self.log_path}:\n{self._log_tail()}")
            try:
                self.agent = GuiRemoteAgent("localhost", self.port, "drp", on_same_machine=True, jpeg_encoding=False,
                                            image_size=None, request_timeout=self.request_timeout)
            except AssertionError:
                raise RuntimeError(f"port {self.port} serves another vlagents agent") from None
            except (ConnectionError, OSError):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no server on port {self.port}:\n{self._log_tail()}") from None
                time.sleep(0.3)
        # the port opens before the model is loaded; a failed initialize() leaves is_initialized False for good
        while not self.agent.is_initialized():
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"server exited ({self.proc.returncode}), log {self.log_path}:\n{self._log_tail()}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"model not loaded after {timeout:.0f} s, log {self.log_path}:\n{self._log_tail()}")
            time.sleep(0.3)
        meta = first_info(self._act(Obs(obs={ROBOT_KEY: SingleObs(info={"drp_query": "meta"})}), 30.0))["meta"]
        if meta.get("ckpt_name") != self.name:
            raise RuntimeError(f"port {self.port} serves {meta.get('ckpt_name')}, expected {self.name}")
        if warmup:  # the first inference pays the CUDA / cuDNN warm-up (150-320 ms)
            self._act(_warmup_obs(meta), 30.0)
        self.meta = meta
        return meta

    def stop(self, timeout: float = 5.0):
        with self.lock:
            if self.agent is not None:
                self.agent.close()
                self.agent = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # rpyc catches KeyboardInterrupt -> agent.close(): shm unregistered
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        self.proc = None

    def _act(self, obs: Obs, timeout: float | None = None) -> Act:
        with self.lock:
            if self.agent is None:
                raise RuntimeError(f"{self.name}: not connected")
            if timeout is not None:
                self.agent.c._config["sync_request_timeout"] = timeout
            try:
                return self.agent.act(obs)
            except (EOFError, ConnectionError, OSError):
                # server gone or restarted: fresh connection and shm segments on the next call
                self.agent.forget_shared_memory()
                try:
                    self.agent.reconnect()
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                if timeout is not None and self.agent is not None and self.agent.c is not None:
                    self.agent.c._config["sync_request_timeout"] = self.request_timeout

    def act(self, obs: Obs) -> Act:
        if not self.ready:
            raise RuntimeError(f"{self.name}: policy server not ready ({self.error or 'starting'})")
        return self._act(obs)


# ---------------------------------------------------------------------------------------------------- the pool
class PolicyPool:
    def __init__(self, root: Path, servers: list[PolicyServer]):
        self.root = Path(root)
        self.info = read_ckpt_info(self.root)
        self.servers = {s.name: s for s in servers}
        self._thread: threading.Thread | None = None

    @classmethod
    def from_dir(cls, root: Path = CKPT_ROOT, base_port: int = BASE_PORT, log_dir: Path | None = None,
                 request_timeout: float = 2.0) -> "PolicyPool":
        root = Path(root)
        info = read_ckpt_info(root)
        dirs = sorted((d for d in root.iterdir() if (d / "model_config.yaml").exists() and (d / "latest.pt").exists()),
                      key=lambda d: (info.get(d.name, {}).get("index", 999), d.name))
        log_dir = Path(log_dir) if log_dir else Path.home() / "drp/logs"
        return cls(root, [PolicyServer(d, base_port + i, log_dir, request_timeout) for i, d in enumerate(dirs)])

    def __getitem__(self, name: str) -> PolicyServer:
        return self.servers[name]

    @property
    def names(self) -> list[str]:
        return list(self.servers)

    def clear_leftovers(self) -> None:
        """Stops vlagents servers still listening on this pool's ports from an earlier panel that did not shut down
        cleanly (they run in their own session, so they outlive a panel that is killed). Only when no other panel is
        running: a running panel's servers are left alone (those checkpoints then fail with the reason)."""
        others = [p for p in _other_panels()]
        for s in self.servers.values():
            owner = _listener(s.port)
            if owner is None:
                continue
            pid, cmd = owner
            if "vlagents" not in cmd or "start-server" not in cmd:
                continue  # not ours: start() reports the port as in use
            if others:
                logger.warning("port %d serves a policy server (pid %d) and another panel is running (pid %s): "
                               "leaving it alone", s.port, pid, others[0])
                continue
            logger.warning("stopping a leftover policy server on port %d (pid %d): %s", s.port, pid, cmd[-90:])
            _stop_pid(pid)

    def start_background(self, first: str | None = None, timeout: float = 180.0, on_progress=None) -> None:
        """Clears leftovers, starts every server, `first` before the others, then waits for all in parallel."""
        self.clear_leftovers()
        order = sorted(self.servers.values(), key=lambda s: s.name != first)

        def one(s: PolicyServer):
            try:
                s.wait_ready(timeout)
                s.error = None
                logger.info("policy server ready: %s (port %d, %s)", s.name, s.port, s.meta["inputs"])
            except Exception as exc:  # noqa: BLE001
                s.error = f"{type(exc).__name__}: {exc}"
                logger.error("policy server %s not ready: %s", s.name, s.error)
            if on_progress is not None:
                on_progress(s.name)

        def run():
            for s in order:
                s.start()
            with ThreadPoolExecutor(max_workers=len(order) or 1) as ex:
                list(ex.map(one, order))
            ready = sum(s.ready for s in order)
            logger.info("policy servers ready: %d/%d", ready, len(order))

        self._thread = threading.Thread(target=run, name="policy-servers", daemon=True)
        self._thread.start()

    def stop_all(self):
        with ThreadPoolExecutor(max_workers=len(self.servers) or 1) as ex:
            list(ex.map(lambda s: s.stop(), self.servers.values()))

    # ------------------------------------------------------------------------------------------ for the panel
    def label(self, name: str) -> str:
        short = self.info.get(name, {}).get("short")
        return f"{short} | {name}" if short else name

    def note(self, name: str) -> str:
        i = self.info.get(name)
        if not i:
            return "-"
        scores = " | ".join(f"{k} {i[k]:.3f}" for k in ("hard", "simple", "ood3x", "ood5x", "collision") if k in i)
        return f"#{i.get('index', '?')} {i.get('role', '')}\nsim: {scores}"

    def describe(self, name: str) -> str:
        s = self.servers[name]
        if s.error:
            return f"{name} | FAILED: {s.error.splitlines()[0]}"
        if not s.ready:
            return f"{name} (policy server starting)"
        m = s.meta
        return f"{name} | {m['inputs']} | epoch {m['epoch']} | chunk {m['chunk_size']} | port {s.port}"

    def inputs(self, name: str) -> str:
        m = self.servers[name].meta
        return m["inputs"] if m else "?"

    def reads(self, name: str) -> set | None:
        """View keys of the inputs the checkpoint reads ("<slot>_rgb", "<slot>_depth"); None while unknown."""
        m = self.servers[name].meta
        if not m:
            return None
        return {f"{s}_rgb" for s in m["rgb_slots"]} | {f"{s}_depth" for s in m["depth_slots"]}

    def goal_is_eef(self, name: str) -> bool:
        m = self.servers[name].meta
        return bool(m and m.get("goal_is_eef"))

    def progress(self) -> str:
        ready = sum(s.ready for s in self.servers.values())
        failed = [s.name for s in self.servers.values() if s.error]
        return f"policy servers ready {ready}/{len(self.servers)}" + (f"; FAILED: {', '.join(failed)}" if failed else "")
