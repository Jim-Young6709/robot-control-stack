"""Single-arm FR3 policy deployment through a vlagents policy server (robots-workstation1).

    policy server (own env, GPU)              this script (RCS venv)
    vlagents AgentService, e.g. "drp"  <--->  RemoteAgent client + RCS env (MuJoCo sim or FR3 hardware)

Station: one FR3 (FCI 192.168.1.12) with a Robotiq 2F-85 (serial DAANTG8W), arm key "right".

Control: ControlMode.JOINTS streamed at `fps` into the torque level joint impedance controller of rcs_fr3
(`async_control=True`, Kp/Kd/torque limit from franka.json). In sim the same controller is emulated
(PD gains, per joint torque limit, 1/fps linear target interpolation, gravity compensated).

Every policy waypoint (absolute joint angles) is shaped before it becomes a target, like the previous DRP
runner (_vet_waypoint + its velocity limited joint stream):
  1. clamp to the FR3 datasheet joint limits minus `joint_limit_margin` (DRP was trained on Panda limits),
  2. cap the jump from the measured joints at `max_waypoint_jump` (scaled, direction kept),
  3. move the commanded target toward it with at most `max_joint_speed` / `max_joint_accel` per joint.
RCS' LimitedAbsoluteAction (max_rel_mov_joints, per joint) stays underneath as a backstop.

Cameras: `camera_source="noise"` sends fresh uniform noise frames (blind test) in the format a real
camera delivers: RGB uint8 HxWx3 under "<slot>", z-depth float32 metres HxW under "<slot>_depth", and
nominal intrinsics in SingleObs.info["intrinsics"]. The policy server does all model specific preprocessing.

Episodes end when every joint is within `goal_tolerance_rad` of the goal, the agent says done, `--max-steps`
is reached, or the env truncates (reflex). In sim the env resets right away; on the robot it holds its pose
until you press 'e' (go home + start) or 'q' (go home).

Usage (from the repo root):
    # sim, blind, headless, 2 episodes of at most 600 steps, trajectory logs in inference_logs/
    .venv/bin/python examples/inference/franka.py --platform sim --headless --max-steps 600 --episodes 2
    # sim with the MuJoCo viewer in real time
    .venv/bin/python examples/inference/franka.py --platform sim --realtime
    # real robot (keys: e=start, r=start+record, s=success/reset, q=stop/reset, o=reload json, x=exit)
    .venv/bin/python examples/inference/franka.py --platform hw
"""

import argparse
import contextlib
import copy
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from time import sleep
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from rcs._core.common import RobotPlatform
from rcs._core.sim import SimConfig
from rcs.camera.interface import BaseCameraSet
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.storage_wrapper import StorageWrapper
from vlagents.client import RemoteAgent
from vlagents.policies.interface import Obs, SingleAct, SingleObs

import rcs

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------
# station constants (hardware specific, not reloadable)
# ----------------------------------------------------------------------------------------------------
ROBOT_KEY = "right"
ROBOT_IP = "192.168.1.12"
# `udevadm info -a -n /dev/ttyUSB0 | grep serial` -> /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAANTG8W-if00-port0
ROBOTIQ_SERIAL = "DAANTG8W"
# ZED wrist camera, only used with camera_source="real"
ZED_CAMERA_DICT: dict[str, str] = {"wrist": "14943057"}

DEFAULT_PLATFORM = RobotPlatform.SIMULATION
CONTROL_MODE = ControlMode.JOINTS
RELATIVETO = RelativeTo.NONE
CONFIG_PATH = Path(__file__).with_suffix(".json")

# FR3 home with the Robotiq mount (flange rotated by 90 deg, see GRIPPER_MOUNT_OFFSETS), same as rcs FR3_DROID
Q_HOME_DEFAULT = rcs.HOME_POSITIONS["FR3_DROID"]
# defaults of the rcs_fr3 joint impedance controller (deoxys joint-impedance-controller.yml)
KP_DEFAULT = [100.0, 100.0, 100.0, 100.0, 75.0, 150.0, 50.0]
KD_DEFAULT = [20.0, 20.0, 20.0, 20.0, 7.5, 15.0, 5.0]
TORQUE_LIMIT_DEFAULT = [5.0] * 7
# FR3 datasheet joint limits (narrower than rcs' MJCF limits and than the Panda limits DRP was trained with)
FR3_Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def d415_intrinsics(width: int = 320, height: int = 240, hfov_deg: float = 54.8, vfov_deg: float = 42.5):
    """Nominal pinhole K of a D415 colour stream at 320x240 (measured FOV 54.8 x 42.5 deg)."""
    fx = width / 2 / np.tan(np.deg2rad(hfov_deg) / 2)
    fy = height / 2 / np.tan(np.deg2rad(vfov_deg) / 2)
    return np.array([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


@dataclass
class InferenceConfig:
    vlagents_host: str = "localhost"
    vlagents_port: int = 20000
    vlagents_model: str = "drp"
    instruction: str = "reach the goal configuration"
    robot_keys: list[str] = field(default_factory=lambda: [ROBOT_KEY])
    # depth frames cannot be jpeg encoded or resized by the client: keep raw frames and let the server resize
    jpeg_encoding: bool = False
    on_same_machine: bool = True
    image_size: tuple[int, int] | None = None
    fps: int = 30
    record_path: str = "inference_recordings"
    n_action_steps: int | None = None
    # backstop of rcs LimitedAbsoluteAction: per joint |target - measured| [rad]
    max_rel_mov_joints: float = 0.35
    max_rel_mov_cart: tuple[float, float] = (0.5, float(np.deg2rad(90)))
    # waypoint shaping (see module docstring)
    joint_limit_margin: float = 0.05
    max_waypoint_jump: float = 0.35
    max_joint_speed: float = 0.5
    max_joint_accel: float = 2.0
    goal_tolerance_rad: float = 0.05
    # episode cap in control steps (1500 at 30 Hz = 50 s, previous DRP runner: 500 chunks at 10 Hz); --max-steps overrides
    max_episode_steps: int | None = 1500
    # "noise": blind test with uniform noise frames, "real": camera frames from the env
    camera_source: str = "noise"
    # per camera slot: width, height, rgb (bool), depth (bool), depth_range_m [lo, hi], optional K (3x3)
    noise_cameras: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            slot: {"width": 320, "height": 240, "rgb": True, "depth": True, "depth_range_m": [0.0, 3.0]}
            for slot in ("front_left", "front_right")
        }
    )
    noise_seed: int | None = None
    # joint goal handed to the policy in SingleObs.info["goal_joints"]; a list of goals is cycled per episode
    goal_joints: list[float] | list[list[float]] | None = None
    # gripper command used when the policy returns no gripper value (1=open, 0=closed)
    default_gripper: float = 1.0
    # False: arm only, no Robotiq in the env (no gripper activation, no gripper obs/actions)
    use_gripper: bool = True
    # hardware: go home with the joint impedance controller (quintic trajectory) instead of rcs' move_home,
    # which uses libfranka's stiff internal motion generator
    impedance_home: bool = True
    # hardware: libfranka collision thresholds 100 Nm/100 N instead of 20/10 while the controller runs
    allow_high_collision: bool = True
    home_max_speed: float = 0.25
    home_min_time: float = 2.0
    home_settle_tol: float = 0.01
    home_settle_timeout: float = 3.0
    q_home: list[float] | None = None
    kp: list[float] = field(default_factory=lambda: list(KP_DEFAULT))
    kd: list[float] = field(default_factory=lambda: list(KD_DEFAULT))
    torque_limit: list[float] = field(default_factory=lambda: list(TORQUE_LIMIT_DEFAULT))
    # sim only: emulate the hardware joint impedance controller instead of the stiff MuJoCo position servos
    sim_emulate_impedance: bool = True
    log_dir: str = "inference_logs"


def load_inference_config() -> InferenceConfig:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(asdict(InferenceConfig()), indent=2) + "\n")
        return InferenceConfig()
    raw = json.loads(CONFIG_PATH.read_text())
    known = InferenceConfig.__dataclass_fields__
    unknown = sorted(set(raw) - set(known))
    if unknown:
        logger.warning("ignoring unknown keys in %s: %s", CONFIG_PATH, unknown)
    return InferenceConfig(**{k: v for k, v in raw.items() if k in known})


def goal_for_episode(cfg: InferenceConfig, episode: int) -> np.ndarray | None:
    if cfg.goal_joints is None:
        return None
    goals = np.asarray(cfg.goal_joints, dtype=np.float64)
    goal = goals if goals.ndim == 1 else goals[episode % len(goals)]
    lo, hi = FR3_Q_MIN + cfg.joint_limit_margin, FR3_Q_MAX - cfg.joint_limit_margin
    if np.any(goal < lo) or np.any(goal > hi):
        message = f"goal {goal.tolist()} is outside the FR3 joint limits (margin {cfg.joint_limit_margin})"
        raise ValueError(message)
    return goal


class WaypointShaper:
    """Policy waypoint -> joint target (FR3 limit clamp, jump cap vs. measured, velocity/accel limit vs. last cmd)."""

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.cmd: np.ndarray | None = None
        self.vel = np.zeros(7)

    def reset(self):
        self.cmd = None
        self.vel = np.zeros(7)

    def __call__(self, waypoint: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        dt = 1.0 / cfg.fps
        w = np.clip(waypoint, FR3_Q_MIN + cfg.joint_limit_margin, FR3_Q_MAX - cfg.joint_limit_margin)
        delta = w - q_meas
        largest = float(np.max(np.abs(delta)))
        if largest > cfg.max_waypoint_jump:
            w = q_meas + delta * (cfg.max_waypoint_jump / largest)
        if self.cmd is None:
            self.cmd = np.array(q_meas, dtype=np.float64)
            self.vel = np.zeros(7)
        d = w - self.cmd
        # largest speed from which we can still stop at w, and no overshoot within one period
        # 0.8: deceleration headroom, the sqrt law alone lags one period and overshoots ~a*dt^2
        v_cap = np.minimum(cfg.max_joint_speed, np.sqrt(2.0 * 0.8 * cfg.max_joint_accel * np.abs(d)))
        v_des = np.sign(d) * np.minimum(v_cap, np.abs(d) / dt)
        self.vel = self.vel + np.clip(v_des - self.vel, -cfg.max_joint_accel * dt, cfg.max_joint_accel * dt)
        self.cmd = self.cmd + self.vel * dt
        return self.cmd.copy()


class TrajectoryLog:
    """Per episode log of measured joints, policy waypoints and commanded targets (saved as npz)."""

    FIELDS = ("t", "q", "dq", "target", "policy_action", "gripper_cmd", "gripper", "latency", "collision")

    def __init__(self, log_dir: str, platform: str):
        self.dir = Path(log_dir)
        self.platform = platform
        self.rows: dict[str, list] = {}
        self.meta: dict[str, Any] = {}

    def start(self, meta: dict[str, Any]):
        self.rows = {k: [] for k in self.FIELDS}
        self.meta = meta

    def add(self, **kw):
        for k, v in kw.items():
            self.rows[k].append(np.asarray(v, dtype=np.float64) if v is not None else np.full(7, np.nan))

    def save(self, tag: str, reason: str) -> Path | None:
        if not self.rows or not self.rows["t"]:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{self.platform}_{tag}.npz"
        meta = dict(self.meta, end_reason=reason)
        np.savez(path, **{k: np.stack(v) for k, v in self.rows.items() if v}, meta=json.dumps(meta, default=str))
        logger.info("saved trajectory log %s (%d steps)", path, len(self.rows["t"]))
        return path


def impedance_home(env: gym.Env, obs: dict, cfg: InferenceConfig, q_home=None, abort_check=None) -> dict:
    """Moves to q_home through the joint impedance controller (env.step with async_control=True).

    A quintic time scaling (zero velocity/acceleration at both ends) is streamed at cfg.fps with a peak joint
    speed of cfg.home_max_speed, then q_home is held until every joint is within cfg.home_settle_tol.
    Returns the last observation.
    """
    if q_home is None:
        q_home = cfg.q_home if cfg.q_home is not None else Q_HOME_DEFAULT
    q_home = np.asarray(q_home, dtype=np.float64)
    key = cfg.robot_keys[0]
    q0 = np.asarray(obs[key]["joints"], dtype=np.float64)
    delta = q_home - q0
    dist = float(np.max(np.abs(delta)))
    duration = max(cfg.home_min_time, 1.875 * dist / cfg.home_max_speed)  # quintic peak speed = 1.875 dist / T
    n = max(1, int(np.ceil(duration * cfg.fps)))
    logger.info("impedance homing: largest joint move %.3f rad over %.1f s", dist, duration)

    def action(q: np.ndarray) -> dict:
        a: dict[str, Any] = {"joints": q}
        if cfg.use_gripper:
            a["gripper"] = np.asarray([1.0], dtype=np.float32)
        return {key: a}

    for k in range(1, n + 1):
        if abort_check is not None and abort_check():
            logger.warning("impedance homing aborted")
            return obs
        tau = k / n
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        obs, _, terminated, truncated, info = env.step(action(q0 + s * delta))
        if terminated or truncated:
            logger.error("impedance homing interrupted (reflex/error): %s", info)
            return obs
    err = float(np.max(np.abs(np.asarray(obs[key]["joints"]) - q_home)))
    t_end = time.perf_counter() + cfg.home_settle_timeout
    while err > cfg.home_settle_tol and time.perf_counter() < t_end:
        obs, _, terminated, truncated, info = env.step(action(q_home))
        if terminated or truncated:
            logger.error("impedance homing interrupted while settling: %s", info)
            return obs
        err = float(np.max(np.abs(np.asarray(obs[key]["joints"]) - q_home)))
    logger.info("impedance homing done, max|q - q_home| = %.4f rad", err)
    return obs


class ModelInference:
    def __init__(self, env: gym.Env, cfg: InferenceConfig, platform: RobotPlatform, max_steps: int | None = None):
        self.env = env
        self._cfg = cfg
        self.platform = platform
        self.is_sim = platform == RobotPlatform.SIMULATION
        self.max_steps = max_steps
        self.max_episodes: int | None = None
        self._episode_running = False
        self._needs_reset = True
        self._command_queue: Queue[str] = Queue()
        self._shutdown_requested = threading.Event()
        self.remote_agent = self._connect(cfg)
        self._action_buffer: list[dict[str, SingleAct]] = []
        self._rng = np.random.default_rng(cfg.noise_seed)
        self.shaper = WaypointShaper(cfg)
        self.episode = 0
        self.step_in_episode = 0
        self.goal: np.ndarray | None = None
        self.log = TrajectoryLog(cfg.log_dir, "sim" if self.is_sim else "hw")

    @staticmethod
    def _connect(cfg: InferenceConfig) -> RemoteAgent:
        while True:
            try:
                agent = RemoteAgent(
                    cfg.vlagents_host,
                    cfg.vlagents_port,
                    cfg.vlagents_model,
                    cfg.on_same_machine,
                    cfg.jpeg_encoding,
                    cfg.image_size,
                )
            except AssertionError:
                message = f"policy server at {cfg.vlagents_host}:{cfg.vlagents_port} does not serve '{cfg.vlagents_model}'"
                raise RuntimeError(message) from None
            except (ConnectionError, OSError) as e:
                logger.info("waiting for policy server %s:%s (%s)", cfg.vlagents_host, cfg.vlagents_port, e)
                sleep(2.0)
                continue
            while not agent.is_initialized():
                logger.info("policy server is still loading the model ...")
                sleep(2.0)
            return agent

    def submit_command(self, command: str) -> None:
        self._command_queue.put(command)

    def request_shutdown(self) -> None:
        self._shutdown_requested.set()

    def _drain_commands(self) -> tuple[bool, bool, bool, bool, bool]:
        start_requested = record_requested = success_requested = stop_requested = reload_requested = False
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                break
            if command == "e":
                start_requested = True
            elif command == "r":
                record_requested = True
            elif command == "s":
                success_requested = True
            elif command == "q":
                stop_requested = True
            elif command == "o":
                reload_requested = True
        return start_requested, record_requested, success_requested, stop_requested, reload_requested

    # ------------------------------------------------------------------ observation / action translation
    def _camera_frames(self, obs: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Returns (frames, intrinsics): RGB under "<slot>", depth [m] under "<slot>_depth", K per slot."""
        cams: dict[str, np.ndarray] = {}
        intrinsics: dict[str, np.ndarray] = {}
        if self._cfg.camera_source == "noise":
            for name, spec in self._cfg.noise_cameras.items():
                h, w = int(spec["height"]), int(spec["width"])
                if spec.get("rgb", True):
                    cams[name] = self._rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
                if spec.get("depth", False):
                    lo, hi = spec.get("depth_range_m", [0.0, 3.0])
                    cams[f"{name}_depth"] = self._rng.uniform(lo, hi, size=(h, w)).astype(np.float32)
                intrinsics[name] = (
                    np.asarray(spec["K"], dtype=np.float32).reshape(3, 3) if "K" in spec else d415_intrinsics(w, h)
                )
            return cams, intrinsics
        # real camera frames from rcs.envs.base.CameraSetWrapper, depth is uint16 scaled by DEPTH_SCALE
        for name, frame in obs.get("frames", {}).items():
            cams[name] = frame["rgb"]["data"]
            if frame["rgb"].get("intrinsics") is not None:
                intrinsics[name] = np.asarray(frame["rgb"]["intrinsics"], dtype=np.float32)[:3, :3]
            if "depth" in frame:
                depth = np.asarray(frame["depth"]["data"]).squeeze(-1)
                cams[f"{name}_depth"] = depth.astype(np.float32) / BaseCameraSet.DEPTH_SCALE
        return cams, intrinsics

    def obs_rcs2agents(self, obs: dict) -> Obs:
        cameras, intrinsics = self._camera_frames(obs)
        obs_by_robot = {}
        for robot in self._cfg.robot_keys:
            info: dict[str, Any] = {"step": self.step_in_episode, "episode": self.episode, "intrinsics": intrinsics}
            if self.goal is not None:
                info["goal_joints"] = self.goal
            obs_by_robot[robot] = SingleObs(
                cameras=cameras if len(self._cfg.robot_keys) == 1 else copy.deepcopy(cameras),
                joints=np.asarray(obs[robot]["joints"], dtype=np.float32),
                gripper=float(np.asarray(obs[robot].get("gripper", [self._cfg.default_gripper])).reshape(-1)[0]),
                xyzrpy=np.asarray(obs[robot]["xyzrpy"], dtype=np.float32) if "xyzrpy" in obs[robot] else None,
                tquat=np.asarray(obs[robot]["tquat"], dtype=np.float32) if "tquat" in obs[robot] else None,
                info=info,
            )
        return Obs(obs=obs_by_robot, language_instruction=self._cfg.instruction)

    def act(self, obs_dict: Obs) -> dict[str, SingleAct]:
        if self._cfg.n_action_steps is None:
            action_chunk = self.remote_agent.act(obs_dict).acts
            if not action_chunk:
                message = "Received empty action chunk from policy"
                raise ValueError(message)
            return action_chunk[0]
        if len(self._action_buffer) == 0:
            action = self.remote_agent.act(obs_dict)
            if not action.acts:
                message = "Received empty action chunk from policy"
                raise ValueError(message)
            self._action_buffer = list(action.acts[: self._cfg.n_action_steps])
        return self._action_buffer.pop(0)

    def action_agents2rcs(self, action: dict[str, SingleAct], obs: dict) -> dict[str, Any]:
        act = {}
        for robot in self._cfg.robot_keys:
            robot_action = action[robot]
            waypoint = np.asarray(robot_action.action, dtype=np.float64)
            if waypoint.shape != (7,) or not np.all(np.isfinite(waypoint)):
                message = f"invalid joint action for {robot}: {waypoint}"
                raise ValueError(message)
            target = self.shaper(waypoint, np.asarray(obs[robot]["joints"], dtype=np.float64))
            act[robot] = {"joints": target}
            if self._cfg.use_gripper:
                gripper = self._cfg.default_gripper if robot_action.gripper is None else float(robot_action.gripper)
                act[robot]["gripper"] = np.asarray([gripper], dtype=np.float32)
        return act

    # ------------------------------------------------------------------ episode handling
    def _reset(self) -> tuple[dict, Obs]:
        self.goal = goal_for_episode(self._cfg, self.episode)
        self.step_in_episode = 0
        obs, _ = self.env.reset()
        if not self.is_sim and self._cfg.impedance_home:
            obs = impedance_home(self.env, obs, self._cfg)
        self._action_buffer = []
        self.shaper.reset()
        self._episode_running = False
        self._needs_reset = False
        return obs, self.obs_rcs2agents(obs)

    def _end_episode(self, reason: str):
        if not self._episode_running:
            return
        logger.info("episode %d ended after %d steps: %s", self.episode, self.step_in_episode, reason)
        self.log.save(f"ep{self.episode:03d}", reason)
        self.episode += 1
        self._episode_running = False
        self._needs_reset = True
        if self.max_episodes is not None and self.episode >= self.max_episodes:
            self.request_shutdown()
        elif self.is_sim:
            self.submit_command("e")
        else:
            logger.info("robot holds its pose. 'e' = go home and start the next episode, 'q' = go home")

    def _start_episode(self, record: bool):
        if isinstance(self.env, StorageWrapper):
            self.env.set_instruction(self._cfg.instruction)
            if record:
                self.env.start_record()
        cfg = self._cfg
        self.log.start(
            {
                "episode": self.episode,
                "goal_joints": None if self.goal is None else self.goal.tolist(),
                "model": cfg.vlagents_model,
                **{
                    k: getattr(cfg, k)
                    for k in (
                        "fps",
                        "camera_source",
                        "max_rel_mov_joints",
                        "max_waypoint_jump",
                        "max_joint_speed",
                        "max_joint_accel",
                        "goal_tolerance_rad",
                        "kp",
                        "kd",
                        "torque_limit",
                        "sim_emulate_impedance",
                    )
                },
            }
        )
        logger.info(
            "starting episode %d%s, goal=%s",
            self.episode,
            " with recording" if record else "",
            None if self.goal is None else np.round(self.goal, 3).tolist(),
        )
        self._episode_running = True

    def loop(self):
        obs, obs_dict = self._reset()
        logger.info(
            "waiting for input: 'e' to start, 'r' to start and record, 's' for success and reset, 'q' to stop and reset, and 'o' to reload config"
        )

        while not self._shutdown_requested.is_set():
            start_requested, record_requested, success_requested, stop_requested, reload_requested = (
                self._drain_commands()
            )

            if reload_requested:
                self._end_episode("config reload")
                self._cfg = load_inference_config()
                self.shaper.cfg = self._cfg
                try:
                    self.remote_agent.reconnect(
                        host=self._cfg.vlagents_host,
                        port=self._cfg.vlagents_port,
                        model=self._cfg.vlagents_model,
                        on_same_machine=self._cfg.on_same_machine,
                        jpeg_encoding=self._cfg.jpeg_encoding,
                        image_size=self._cfg.image_size,
                    )
                    logger.info(
                        "reloaded config from %s with host=%s port=%s model=%s",
                        CONFIG_PATH,
                        self._cfg.vlagents_host,
                        self._cfg.vlagents_port,
                        self._cfg.vlagents_model,
                    )
                except Exception:
                    logger.exception("failed to reconnect after reloading %s", CONFIG_PATH)
                if isinstance(self.env, StorageWrapper):
                    self.env.base_dir = self._cfg.record_path
                    self.env.set_instruction(self._cfg.instruction)
                obs, obs_dict = self._reset()

            if success_requested:
                self._end_episode("success")
                self.env.get_wrapper_attr("success")()
                obs, obs_dict = self._reset()

            if stop_requested:
                self._end_episode("stopped")
                obs, obs_dict = self._reset()

            if self._shutdown_requested.is_set():
                break

            if not self._episode_running:
                try:
                    self.remote_agent.ensure_connected()
                except Exception:
                    sleep(0.5)
                    continue
                if start_requested or record_requested:
                    if self._needs_reset:
                        obs, obs_dict = self._reset()
                    self._start_episode(record_requested)
                else:
                    sleep(0.05)
                    continue

            t0 = time.perf_counter()
            try:
                # RemoteAgent replaces the frames in place by shared memory / jpeg payloads
                action = self.act(copy.deepcopy(obs_dict))
                a = self.action_agents2rcs(action, obs)
            except Exception:
                logger.exception("policy failed, stopping episode")
                self._end_episode("policy error")
                if self.is_sim:
                    self.request_shutdown()
                continue
            latency = time.perf_counter() - t0

            if any(robot_action.done for robot_action in action.values()):
                self._end_episode("done issued by agent")
                continue

            obs, _, terminated, truncated, info = self.env.step(a)
            if terminated or truncated:
                # rcs_fr3 FR3HW returns truncated=True after a FrankaControlException (reflex) and recovers errors
                self._end_episode(f"env terminated={terminated} truncated={truncated} info={info}")
                continue

            for robot in self._cfg.robot_keys:
                rinfo = info.get(robot, {})
                dq = None
                if "robot_state" in obs[robot]:
                    dq = obs[robot]["robot_state"]["dq"]
                elif "dq" in rinfo:
                    dq = rinfo["dq"]
                self.log.add(
                    t=time.time(),
                    q=obs[robot]["joints"],
                    dq=dq,
                    target=rinfo.get("limited_absolute_action", a[robot]["joints"]),
                    policy_action=action[robot].action,
                    gripper_cmd=a[robot].get("gripper", [np.nan]),
                    gripper=np.asarray(obs[robot].get("gripper", [np.nan]), dtype=np.float64).reshape(-1)[:1],
                    latency=[latency],
                    collision=[float(bool(rinfo.get("collision", False)))],
                )
            self.step_in_episode += 1
            obs_dict = self.obs_rcs2agents(obs)

            if self.goal is not None:
                err = np.abs(np.asarray(obs[self._cfg.robot_keys[0]]["joints"]) - self.goal)
                if float(err.max()) <= self._cfg.goal_tolerance_rad:
                    self._end_episode(f"goal reached (max joint error {err.max():.3f} rad)")
                    continue
            if self.max_steps is not None and self.step_in_episode >= self.max_steps:
                self._end_episode(f"max_steps={self.max_steps} reached")


def command_loop(controller: ModelInference) -> None:
    prompt = "Command [e=start, r=record, s=success/reset, q=stop/reset, o=reload, x=exit]: "
    while True:
        try:
            command = input(prompt).strip().lower()
        except EOFError:
            command = "x"
        except KeyboardInterrupt:
            print()
            command = "x"

        if not command:
            continue
        if command == "x":
            controller.request_shutdown()
            return
        if command in {"e", "r", "s", "q", "o"}:
            controller.submit_command(command)
            continue
        logger.info("unknown command %r", command)


# ----------------------------------------------------------------------------------------------------
# sim: emulation of the rcs_fr3 joint impedance controller
# ----------------------------------------------------------------------------------------------------
class JointImpedanceSimEmulation(gym.Wrapper):
    """Makes the MuJoCo arm behave like rcs_fr3's torque level joint impedance controller (Franka::joint_controller).

    - PD gains Kp/Kd on the position actuators instead of the stiff MuJoCo servos (set via SimRobotConfig.kp/kv),
    - per joint limit on the PD torque (TorqueSafetyGuardFn); gravity is compensated on top, like on the robot,
    - every new target is linearly interpolated from the previous target over one control period
      (LinearJointPositionTrajInterpolator with traj_interpolation_time_fraction=1).
    """

    def __init__(self, env: gym.Env, robot_key: str, torque_limit: list[float]):
        super().__init__(env)
        self._sim = env.get_wrapper_attr("sim")
        robot = env.get_wrapper_attr("robot")[robot_key]
        rcfg = robot.get_config()
        m = self._sim.model
        self._act_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in rcfg.actuators], dtype=np.int64
        )
        jnt_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in rcfg.joints]
        self._dof_ids = np.array([m.jnt_dofadr[j] for j in jnt_ids], dtype=np.int64)
        limit = np.asarray(torque_limit, dtype=np.float64)
        # Limit only the PD (actuator) force. The FR3 joints use actuatorgravcomp, i.e. gravity compensation is added
        # on top of the actuator force and only bounded by the joint's actuatorfrcrange (87/12 Nm), which mirrors the
        # robot: libfranka compensates gravity internally and TorqueSafetyGuardFn clamps only the commanded torque.
        for a, lim in zip(self._act_ids, limit, strict=True):
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = [-lim, lim]
        self._prev_target: np.ndarray | None = None
        self._lead_env = env.get_wrapper_attr("lead_env")
        self._lead_env.step_sim = self._step_sim_interpolated
        logger.info(
            "sim emulates joint impedance: kp=%s kv=%s torque_limit=%s",
            m.actuator_gainprm[self._act_ids, 0].round(2).tolist(),
            (-m.actuator_biasprm[self._act_ids, 2]).round(2).tolist(),
            limit.tolist(),
        )

    def _step_sim_interpolated(self):
        cfg = self._sim.get_config()
        n = max(1, round(1 / cfg.frequency / self._sim.model.opt.timestep))
        new = self._sim.data.ctrl[self._act_ids].copy()
        prev = new if self._prev_target is None else self._prev_target
        for i in range(1, n + 1):
            self._sim.data.ctrl[self._act_ids] = prev + (i / n) * (new - prev)
            self._sim.step(1)
        self._prev_target = new
        if cfg.realtime:
            self._lead_env.frame_rate.frame_rate = cfg.frequency
            self._lead_env.frame_rate()

    def dq(self) -> np.ndarray:
        return self._sim.data.qvel[self._dof_ids].copy()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in info:
            if isinstance(info[key], dict):
                info[key]["dq"] = self.dq()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self._prev_target = self._sim.data.ctrl[self._act_ids].copy()
        return ret


def get_sim_env(cfg: InferenceConfig, headless: bool, realtime: bool) -> gym.Env:
    from rcs.envs.configs import EmptyWorldDroid

    # FR3 + Robotiq 2F-85 under the key "right", the same arm/gripper combination as on this station
    scene = EmptyWorldDroid()
    sim_cfg = scene.config()
    sim_cfg.sim_cfg = SimConfig(async_control=True, realtime=realtime, frequency=cfg.fps, max_convergence_steps=500)
    # blind: no rendered cameras, frames come from the noise generator
    sim_cfg.camera_cfgs = None
    sim_cfg.camera_adds = None
    sim_cfg.robot_frame_objects = None
    sim_cfg.control_mode = CONTROL_MODE
    sim_cfg.relative_to = RELATIVETO
    sim_cfg.max_relative_movement = cfg.max_rel_mov_joints
    sim_cfg.wrapper_cfg.binary_gripper = True
    sim_cfg.wrapper_cfg.include_depth = False
    sim_cfg.headless = headless
    if not cfg.use_gripper:
        sim_cfg.gripper_cfgs = None
    robot_cfg = sim_cfg.robot_cfgs[ROBOT_KEY]
    robot_cfg.q_home = np.asarray(cfg.q_home if cfg.q_home is not None else Q_HOME_DEFAULT, dtype=np.float64)
    if cfg.sim_emulate_impedance:
        robot_cfg.kp = list(cfg.kp)
        robot_cfg.kv = list(cfg.kd)
    env = scene.create_env(sim_cfg)
    if cfg.sim_emulate_impedance:
        env = JointImpedanceSimEmulation(env, ROBOT_KEY, cfg.torque_limit)
    return env


def get_hw_env(cfg: InferenceConfig) -> gym.Env:
    from rcs_fr3.configs import DefaultFR3HardwareEnv
    from rcs_fr3.creators import (
        FR3MultiHardwareEnvCreatorConfig,
        HardwareCameraCreatorConfig,
        RCSFR3MultiConfigEnvCreator,
    )
    from rcs_robotiq2f85.hw import RobotiQ2F85GripperConfig

    base = DefaultFR3HardwareEnv()
    base.ip = ROBOT_IP
    robot_cfg = base.config().robot_cfg
    # torque level joint impedance controller (Franka::joint_controller) fed by streamed joint targets
    robot_cfg.async_control = True
    robot_cfg.kp = np.asarray(cfg.kp, dtype=np.float64)
    robot_cfg.kd = np.asarray(cfg.kd, dtype=np.float64)
    robot_cfg.torque_limit = np.asarray(cfg.torque_limit, dtype=np.float64)
    # as in examples/fr3/fr3_env_cartesian_control.py on this station: 100 Nm / 100 N collision thresholds while
    # the controller runs (fewer spurious reflexes under torque control); the 5 Nm torque cap + e-stop are the limit
    robot_cfg.allow_high_collision = cfg.allow_high_collision
    # on (re)start of the controller, ramp from the measured pose to the first target before streaming
    robot_cfg.blocking_move_on_start = True
    # interpolation window of the controller must match the rate at which we stream actions
    robot_cfg.policy_rate = cfg.fps
    # no PREEMPT_RT kernel on this workstation
    robot_cfg.ignore_realtime = True
    # only used by rcs move_home, i.e. when impedance_home is false
    robot_cfg.speed_factor = 0.1
    robot_cfg.tcp_offset = rcs.GRIPPER_TCP_OFFSETS[rcs.common.GripperType("Robotiq2F85")]
    robot_cfg.q_home = np.asarray(cfg.q_home if cfg.q_home is not None else Q_HOME_DEFAULT, dtype=np.float64)

    camera_cfgs = None
    if cfg.camera_source == "real" and ZED_CAMERA_DICT:
        camera_cfgs = {
            "zed": HardwareCameraCreatorConfig(
                camera_type_id="zed",
                camera_cfgs={
                    name: rcs.common.BaseCameraConfig(
                        identifier=identifier, resolution_width=1280, resolution_height=720, frame_rate=30
                    )
                    for name, identifier in ZED_CAMERA_DICT.items()
                },
                kwargs={"enable_depth": True, "enable_imu": False},
            )
        }

    hw_cfg = FR3MultiHardwareEnvCreatorConfig(
        robot_cfgs={ROBOT_KEY: robot_cfg},
        control_mode=CONTROL_MODE,
        gripper_cfgs=(
            {ROBOT_KEY: RobotiQ2F85GripperConfig(serial_number=ROBOTIQ_SERIAL, speed=100, force=50, async_control=True)}
            if cfg.use_gripper
            else None
        ),
        camera_cfgs=camera_cfgs,
        max_relative_movement=cfg.max_rel_mov_joints,
        relative_to=RELATIVETO,
        robot_to_shared_base_frame={ROBOT_KEY: rcs.common.Pose()},
        frequency=cfg.fps,
    )
    hw_cfg.wrapper_cfg.binary_gripper = True
    hw_cfg.wrapper_cfg.include_depth = camera_cfgs is not None
    # with impedance_home the reset only recovers errors, ModelInference/impedance_home moves home
    hw_cfg.wrapper_cfg.home_on_reset = not cfg.impedance_home
    return RCSFR3MultiConfigEnvCreator().create_env(hw_cfg)


def get_env(cfg: InferenceConfig, platform: RobotPlatform, headless: bool, realtime: bool) -> gym.Env:
    env = get_hw_env(cfg) if platform == RobotPlatform.HARDWARE else get_sim_env(cfg, headless, realtime)
    return StorageWrapper(
        env,
        cfg.record_path,
        cfg.instruction,
        batch_size=32,
        max_rows_per_group=2,
        max_rows_per_file=10,
        allow_wrapper_instruction=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", choices=["sim", "hw"], default=None, help="default: DEFAULT_PLATFORM")
    parser.add_argument("--headless", action="store_true", help="sim: no MuJoCo viewer")
    parser.add_argument("--realtime", action="store_true", help="sim: run at wall clock speed")
    parser.add_argument("--max-steps", type=int, default=None, help="override max_episode_steps of the config")
    parser.add_argument("--episodes", type=int, default=None, help="exit after this many episodes")
    parser.add_argument("--config", default=None, help="json config (default: franka.json next to this script)")
    return parser.parse_args()


def main():
    global CONFIG_PATH
    args = parse_args()
    if args.config is not None:
        CONFIG_PATH = Path(args.config).expanduser().resolve()
    if args.platform is None:
        platform = DEFAULT_PLATFORM
    else:
        platform = RobotPlatform.HARDWARE if args.platform == "hw" else RobotPlatform.SIMULATION
    cfg = load_inference_config()
    goal_for_episode(cfg, 0)  # validate the goals before anything moves
    logger.info("config: %s", json.dumps(asdict(cfg)))
    if platform == RobotPlatform.HARDWARE and cfg.use_gripper:
        input(
            f"[HW] connecting to the FR3 at {ROBOT_IP} (no arm motion) and ACTIVATING the Robotiq {ROBOTIQ_SERIAL}: "
            f"the fingers move during activation. Press Enter to continue, Ctrl-C to abort."
        )
    env_rel = get_env(cfg, platform, args.headless, args.realtime)
    max_steps = args.max_steps if args.max_steps is not None else cfg.max_episode_steps
    controller = ModelInference(env_rel, cfg, platform, max_steps=max_steps)
    controller.max_episodes = args.episodes
    with env_rel, contextlib.closing(controller.remote_agent):
        if platform == RobotPlatform.SIMULATION:
            # sim doesnt allow threads (greenlets), run the loop in the main thread and start right away
            controller.submit_command("e")  # use "r" to record the sim rollout
            try:
                controller.loop()
            except KeyboardInterrupt:
                controller._end_episode("keyboard interrupt")
        else:
            q_home = cfg.q_home if cfg.q_home is not None else Q_HOME_DEFAULT.tolist()
            how = (
                f"with the joint impedance controller (<= {cfg.home_max_speed} rad/s)"
                if cfg.impedance_home
                else "with rcs move_home (libfranka motion generator, speed_factor 0.1)"
            )
            input(
                f"[HW] the robot will now move to its home pose {np.round(q_home, 3).tolist()} {how}. "
                f"Keep the e-stop in reach. Press Enter to continue, Ctrl-C to abort."
            )
            worker = threading.Thread(target=controller.loop, name="model-inference", daemon=True)
            worker.start()
            command_loop(controller)
            worker.join()


if __name__ == "__main__":
    main()
