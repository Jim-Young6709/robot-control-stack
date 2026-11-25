"""The panel's controller: one worker thread owns the RCS env and steps it at the control rate (cfg.fps, 30 Hz); the
viser callbacks only change state, goal and settings here, and read snapshots the worker publishes.

Every step sends a joint target to rcs_fr3's joint impedance controller (async_control) through the RCS env, and the
current gripper target through the RCS GripperWrapper:

* idle / reached: the hold target (after Stop: the stopping point of the current motion, then held),
* running: the active checkpoint's waypoint (a new chunk from its vlagents server every `exec_steps` waypoints, a new
  waypoint every 1/exec_rate s), shaped by franka.WaypointShaper (FR3 limit clamp, jump cap vs. the measured joints,
  velocity / acceleration limit) at the control rate,
* resetting: a quintic trajectory to the home pose (peak speed home_max_speed), then held until settled,
* error: nothing is sent; the robot holds its last target (or libfranka holds it after a reflex) until Recover.
"""

import enum
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from rcs._core.common import RobotPlatform
from rcs.envs.base import CameraSetWrapper, GripperWrapper, MultiRobotWrapper

import franka

import rcs

from .policies import chunk_of, make_obs, rcs_frames_to_cameras

logger = logging.getLogger(__name__)
KEY = franka.ROBOT_KEY

#: home poses: RCS's FR3 home (also franka.py's), the previous panel's DROID reset pose, the libfranka ready pose
HOME_POSES = {
    "RCS home (FR3_DROID)": np.asarray(rcs.HOME_POSITIONS["FR3_DROID"], dtype=np.float64),
    "DROID reset pose": np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0]),
    "Franka ready pose": np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]),
}
DEFAULT_HOME = "RCS home (FR3_DROID)"


class State(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    RESETTING = "resetting"
    REACHED = "reached"
    ERROR = "error"


@dataclass
class RunConfig:
    """Live settings (the GUI writes them, the worker reads them every step). franka.WaypointShaper reads fps,
    joint_limit_margin, max_waypoint_jump, max_joint_speed and max_joint_accel from here."""

    fps: float = 30.0
    exec_steps: int = 1
    exec_rate_hz: float = 30.0
    max_waypoint_jump: float = 0.35
    max_joint_speed: float = 0.5
    max_joint_accel: float = 2.0
    joint_limit_margin: float = 0.05
    goal_tolerance_rad: float = 0.05
    goal_tolerance_m: float = 0.02
    goal_tolerance_deg: float = 5.0
    max_chunks: int = 500
    continuous: bool = False
    alternate_goals: bool = False
    noise_slots: set = field(default_factory=set)
    home_max_speed: float = 0.25
    home_min_time: float = 2.0
    home_settle_tol: float = 0.01
    home_settle_timeout: float = 3.0
    #: a rollout stops when the camera frames are older than this [s]
    max_frame_age_s: float = 0.5


@dataclass
class RunStats:
    chunks: int = 0
    waypoints: int = 0
    last_inference_ms: float = float("nan")
    last_chunk_max_step: float = float("nan")
    clamped_waypoints: int = 0
    started_at: float | None = None


def _franka_errors() -> tuple:
    try:
        from rcs_fr3._core.hw import exceptions as ex
    except ImportError:
        return ()
    return tuple(getattr(ex, n) for n in ("FrankaControlException", "FrankaCommandException", "FrankaNetworkException",
                                         "FrankaProtocolException", "FrankaInvalidOperationException",
                                         "FrankaRealtimeException", "FrankaException") if hasattr(ex, n))


def _find_wrapper(env, cls):
    e = env
    while e is not None:
        if isinstance(e, cls):
            return e
        if isinstance(e, MultiRobotWrapper):
            e = e.envs[KEY]
        else:
            e = getattr(e, "env", None)
    return None


class PanelRunner:
    def __init__(self, make_env, platform: RobotPlatform, pool, active_ckpt: str, kin, config: RunConfig,
                 depth_mask, camera_slots=("front_left", "front_right")):
        self._make_env = make_env
        self.platform = platform
        self.pool = pool
        self.kin = kin
        self.config = config
        self.depth_mask = depth_mask
        self.camera_slots = list(camera_slots)
        self.control_hz = float(config.fps)
        self.stats = RunStats()

        self._lock = threading.RLock()
        self._state = State.IDLE
        self._message = "starting"
        self._goal: np.ndarray | None = None
        self.goal_slots: dict = {"A": None, "B": None}
        self.active_slot: str | None = None
        self.laps = 0
        self._active = active_ckpt

        self._jobs: list = []  # (callable(env) -> str, Event, result holder), run on the worker thread
        self._shutdown = threading.Event()
        self.ready = threading.Event()
        self.startup_error: str | None = None
        self._snap: dict = {"q": None, "mode": "connecting", "motion_ready": False, "gripper": None}
        self._cam_snap: dict | None = None
        self._frames: dict = {}
        self._intrinsics: dict = {}
        self._frames_ts: float | None = None
        self._frames_seen_at = 0.0

        self._shaper = franka.WaypointShaper(config)
        self._hold: np.ndarray | None = None
        self._q: np.ndarray | None = None
        self._grip_width = 1.0  # RCS: 1 = open, 0 = closed
        self._grip_speed_force = (150.0, 104.3)
        self._grip_dirty = False
        self._grip_activating = False
        self._chunk: np.ndarray | None = None
        self._chunk_idx = 0
        self._wp_tick = 0
        self._step_count = 0
        self._episode = 0
        self._home: dict | None = None
        self._prev_state = State.IDLE
        self._rng = np.random.default_rng(0)
        self._errors = _franka_errors()
        self._thread = threading.Thread(target=self._run, name="rcs-env", daemon=True)

    # ======================================================================================= GUI-facing API
    def start(self) -> None:
        self._thread.start()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _set_state(self, state: State, message: str = "") -> None:
        with self._lock:
            if self._state != state:
                logger.info("runner: %s -> %s%s", self._state.value, state.value, f" ({message})" if message else "")
            self._state = state
            self._message = message

    @property
    def goal(self):
        with self._lock:
            return None if self._goal is None else self._goal.copy()

    def _clamp(self, q) -> np.ndarray:
        m = self.config.joint_limit_margin
        return np.clip(np.asarray(q, dtype=np.float64).reshape(7), franka.FR3_Q_MIN + m, franka.FR3_Q_MAX - m)

    def set_goal(self, q) -> np.ndarray:
        q = self._clamp(q)
        with self._lock:
            self._goal = q
        return q

    def save_goal_slot(self, slot: str, q) -> np.ndarray:
        q = self._clamp(q)
        with self._lock:
            self.goal_slots[slot] = q
        return q

    def use_goal_slot(self, slot: str) -> str:
        with self._lock:
            q = self.goal_slots.get(slot)
            if q is None:
                return f"slot {slot} is empty"
            self._goal = q.copy()
            self.active_slot = slot
        return f"goal set from slot {slot}"

    def activate(self) -> str:
        with self._lock:
            if self._state is State.RUNNING:
                return "already running"
            if self._goal is None:
                return "refused: no goal set"
            if self._state is State.RESETTING:
                return "refused: homing in progress"
            if self._state is State.ERROR:
                return "refused: robot in error - press Recover first"
            if not self._snap.get("motion_ready"):
                return f"refused: robot mode {self._snap.get('mode')}, not motion ready"
            if not self.pool[self._active].ready:
                return f"refused: policy server of {self._active} not ready ({self.pool.describe(self._active)})"
            self.stats = RunStats(started_at=time.time())
            self._chunk = None
            self._episode += 1
            self._step_count = 0
            self._set_state(State.RUNNING)
        return "running"

    def stop(self) -> str:
        with self._lock:
            if self._state in (State.RUNNING, State.RESETTING, State.REACHED):
                self._set_state(State.IDLE, "stopped by user")
        return "stopped"

    def reset_home(self, q_home) -> str:
        with self._lock:
            if self._state is State.RESETTING:
                return "already homing"
            if self._state is State.ERROR:
                return "refused: robot in error - press Recover first"
            if not self._snap.get("motion_ready"):
                return f"refused: robot mode {self._snap.get('mode')}, not motion ready"
            self._home = {"target": self._clamp(q_home), "t0": None}
            self._set_state(State.RESETTING)
        return "homing"

    def clear_error(self) -> str:
        """Recovers the FR3 (stop controller, automatic error recovery) on the worker thread; then holds the pose."""

        def job(env) -> str:
            if self.platform == RobotPlatform.HARDWARE:
                env.get_wrapper_attr("robot")[KEY].reset()
            self._resync_hold()
            self._set_state(State.IDLE, "error cleared")
            return "cleared"

        return self._run_job(job, timeout=15.0)

    def set_policy(self, name: str) -> str:
        """Returns "" on success, else why the switch is refused."""
        if name not in self.pool.servers:
            return f"REFUSED: unknown checkpoint {name}"
        if not self.pool[name].ready:
            return f"REFUSED: {self.pool.describe(name)}"
        with self._lock:
            if self._state is State.RUNNING:
                self._set_state(State.IDLE, "checkpoint switched")
            self._active = name
            self._chunk = None
        return ""

    @property
    def active_checkpoint(self) -> str:
        with self._lock:
            return self._active

    def gripper_activate(self) -> str:
        if self._grip_activating:
            return "activation already running"
        try:
            gripper = self._gripper
        except AttributeError:
            return "no gripper"
        if gripper is None:
            return "no gripper"
        if not hasattr(gripper, "activate"):  # sim gripper: nothing to activate
            return "activated (sim)"

        def run():
            self._grip_activating = True
            try:
                gripper.activate()  # blocking, the fingers sweep; the arm keeps being held meanwhile
                self._grip_dirty = True  # re-send the current target once activated
                logger.info("gripper activated")
            except Exception:  # noqa: BLE001
                logger.exception("gripper activation failed")
            finally:
                self._grip_activating = False

        threading.Thread(target=run, name="gripper-activate", daemon=True).start()
        return "activating (the fingers move)..."

    def set_gripper(self, closure: float, speed: int = 255, force: int = 100) -> None:
        """closure 0 = open .. 1 = closed; speed/force on the Robotiq's 0-255 scale (-> 20-150 mm/s, 20-235 N)."""
        self._grip_width = float(np.clip(1.0 - closure, 0.0, 1.0))
        self._grip_speed_force = (20.0 + speed * 130.0 / 255.0 + 1e-9, 20.0 + force * 215.0 / 255.0 + 1e-9)
        self._grip_dirty = True

    # ------------------------------------------------------------------- depth post-processing (cameras)
    def _realsense_sets(self) -> list:
        cam_set = getattr(self, "_camera_set", None)
        return [c for c in getattr(cam_set, "cameras", []) if hasattr(c, "set_depth_processing")]

    @property
    def has_realsense(self) -> bool:
        return bool(self._realsense_sets())

    def depth_processing(self) -> dict:
        """Camera-side post-processing flags (disparity, spatial, temporal, hole_filling), emitter and the chain
        description; {} without RealSense cameras (sim, --blind)."""
        sets = self._realsense_sets()
        return sets[0].depth_processing() if sets else {}

    def set_depth_processing(self, **flags: bool) -> None:
        """Switches steps on every camera while streaming (rcs_realsense applies them from the next frame)."""
        for cam in self._realsense_sets():
            cam.set_depth_processing(**flags)

    def set_emitter(self, on: bool) -> None:
        for cam in self._realsense_sets():
            cam.enable_emitter(bool(on))

    def snapshot(self) -> dict:
        with self._lock:
            state, message, goal = self._state, self._message, self._goal
            snap = dict(self._snap)
        q = snap.get("q")
        return {
            "state": state.value, "message": message, "q": q, "goal": None if goal is None else goal.copy(),
            "dist": None if (goal is None or q is None) else float(np.max(np.abs(goal - q))),
            "robot_mode": snap.get("mode"), "motion_ready": snap.get("motion_ready"), "stats": self.stats,
            "gripper": snap.get("gripper") or {"closure": None, "text": "-"},
        }

    def camera_snapshot(self) -> dict | None:
        with self._lock:
            return self._cam_snap

    def shutdown(self, timeout: float = 15.0) -> None:
        self._shutdown.set()
        self._thread.join(timeout=timeout)

    # ======================================================================================= worker thread
    def _run_job(self, job, timeout: float) -> str:
        done, result = threading.Event(), {}
        with self._lock:
            self._jobs.append((job, done, result))
        if not done.wait(timeout):
            return "timed out (still running on the robot thread)"
        return result.get("value", "?")

    def _run(self) -> None:
        try:
            self.env = self._make_env()
            obs, _ = self.env.reset()
            self._robot = self.env.get_wrapper_attr("robot")[KEY]
            try:
                self._gripper = self.env.get_wrapper_attr("gripper")[KEY]
            except AttributeError:
                self._gripper = None
            self._grip_wrapper = _find_wrapper(self.env, GripperWrapper)
            self._cam_wrapper = _find_wrapper(self.env, CameraSetWrapper)
            self._camera_set = self._cam_wrapper.camera_set if self._cam_wrapper is not None else None
            self._ingest(obs, {})
            if self._gripper is not None:  # start from where the fingers are: nothing moves on startup
                self._grip_width = float(np.clip(self._gripper.get_normalized_width(), 0.0, 1.0))
            self._resync_hold()
            self._set_state(State.IDLE, "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("could not create the RCS env")
            self.startup_error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
            return
        self.ready.set()
        try:
            while not self._shutdown.is_set():
                try:
                    self._tick()
                except Exception as exc:  # noqa: BLE001 - never let the worker die
                    logger.exception("runner step failed")
                    self._set_state(State.ERROR, f"{type(exc).__name__}: {exc}")
                    time.sleep(0.2)
            self._wind_down()
        finally:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                logger.exception("env.close failed")
            self.env = self._robot = self._gripper = None
            logger.info("RCS env closed")

    def _tick(self) -> None:
        with self._lock:
            jobs, self._jobs = self._jobs, []
        for job, done, result in jobs:
            try:
                result["value"] = job(self.env)
            except Exception as exc:  # noqa: BLE001
                logger.exception("robot job failed")
                result["value"] = f"failed: {type(exc).__name__}: {exc}"
            done.set()

        state = self.state
        if state is not self._prev_state:
            self._on_transition(self._prev_state, state)
            self._prev_state = state
        if state is State.ERROR:
            self._observe_only()
            time.sleep(1.0 / self.control_hz)
        elif state is State.RUNNING:
            self._policy_step()
        elif state is State.RESETTING:
            self._home_step()
        else:
            self._hold_step()
            self._maybe_preview()

    def _hold_step(self) -> None:
        sh = self._shaper
        if sh.cmd is None or (float(np.max(np.abs(sh.vel))) < 1e-4 and np.allclose(sh.cmd, self._hold, atol=1e-4)):
            # at rest on the hold pose: send it as is (the shaper would pull a pose outside the margin inward)
            sh.cmd, sh.vel = self._hold.copy(), np.zeros(7)
            self._step(self._hold)
        else:
            self._step(sh(self._hold, self._q))  # still decelerating onto it

    def _on_transition(self, old: State, new: State) -> None:
        if old in (State.RUNNING, State.RESETTING) and new in (State.IDLE, State.REACHED):
            # decelerate to the stopping point of the current command, then hold it
            cmd, vel = self._shaper.cmd, self._shaper.vel
            if cmd is not None:
                a = max(self.config.max_joint_accel, 1e-3)
                self._hold = self._clamp(cmd + vel * np.abs(vel) / (2.0 * a))
            else:
                self._hold = self._q.copy()
        if new is State.RUNNING:
            self._chunk, self._chunk_idx, self._wp_tick = None, 0, 0

    def _resync_hold(self) -> None:
        q = self._read_q()
        self._q = q
        self._hold = q.copy()
        self._shaper.reset()

    def _read_q(self) -> np.ndarray:
        if self.platform == RobotPlatform.HARDWARE:
            return np.asarray(self._robot.get_state().robot_state.q, dtype=np.float64)
        return np.asarray(self._robot.get_joint_position(), dtype=np.float64)

    # ----------------------------------------------------------------------------------------- stepping
    def _action(self, joints) -> dict:
        a = {"joints": np.asarray(joints, dtype=np.float64)}
        if self._gripper is not None:
            if self._grip_dirty and self._grip_wrapper is not None:
                self._grip_dirty = False
                gcfg = self._gripper.get_config()
                if hasattr(gcfg, "speed"):
                    gcfg.speed, gcfg.force = self._grip_speed_force
                self._grip_wrapper._last_gripper_cmd = None  # re-send even if the width did not change
            a["gripper"] = np.asarray([self._grip_width], dtype=np.float32)
        return {KEY: a}

    def _step(self, joints) -> dict | None:
        """One env.step. On a robot error: ERROR state, nothing more is sent until Recover."""
        try:
            obs, _, terminated, truncated, info = self.env.step(self._action(joints))
        except self._errors as exc:
            logger.error("robot error: %s", exc)
            self._set_state(State.ERROR, f"robot error: {type(exc).__name__}: {exc}. Press Recover.")
            return None
        if terminated or truncated:
            # FR3HW caught a reflex and already ran the automatic recovery; do not resume on our own
            self._set_state(State.ERROR, "reflex / control error (see log). Press Recover to re-arm.")
            return None
        self._ingest(obs, info)
        return obs

    def _ingest(self, obs: dict, info: dict) -> None:
        self._q = np.asarray(obs[KEY]["joints"], dtype=np.float64)
        if self.platform == RobotPlatform.HARDWARE:
            mode = self._robot.get_state().robot_state.robot_mode.name.removeprefix("k")
        else:
            mode = "Sim"
        grip = None
        if self._gripper is not None:
            width = float(obs[KEY].get("gripper", [np.nan])[0])
            active = self._gripper.is_activated() if hasattr(self._gripper, "is_activated") else True
            grip = {"closure": 1.0 - self._grip_width,
                    "text": f"opening {width * 85:.0f} mm ({1 - width:.2f} closed) | "
                            + ("activating..." if self._grip_activating else f"active={active}")}
        frames = obs.get("frames")
        if frames:
            self._frames, self._intrinsics = rcs_frames_to_cameras(frames)
            ts = info.get("frame_timestamp") if info else None
            if ts != self._frames_ts:
                self._frames_ts, self._frames_seen_at = ts, time.monotonic()
        with self._lock:
            self._snap = {"q": self._q.copy(), "mode": mode, "motion_ready": mode in ("Idle", "Move", "Sim"),
                          "gripper": grip}

    def _observe_only(self) -> None:
        """ERROR state: read state and camera frames without commanding anything."""
        try:
            q = self._read_q()
            mode = (self._robot.get_state().robot_state.robot_mode.name.removeprefix("k")
                    if self.platform == RobotPlatform.HARDWARE else "Sim")
            with self._lock:
                self._snap = dict(self._snap, q=q, mode=mode, motion_ready=False)
            self._q = q
            if self._cam_wrapper is not None:
                obs, _ = self._cam_wrapper.observation({}, {})
                if obs.get("frames"):
                    self._frames, self._intrinsics = rcs_frames_to_cameras(obs["frames"])
            self._maybe_preview()
        except Exception:  # noqa: BLE001
            logger.debug("observe failed", exc_info=True)

    def _noise_frames(self):
        """Sim has no cameras: noise in the format of the real ones (blind), nominal D415 K."""
        h, w = 240, 320
        frames, K = {}, {}
        for slot in self.camera_slots:
            frames[slot] = self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            frames[f"{slot}_depth"] = self._rng.uniform(0.0, 3.0, (h, w)).astype(np.float32)
            K[slot] = franka.d415_intrinsics(w, h)
        return frames, K

    def _current_frames(self) -> dict:
        if self._cam_wrapper is None:  # sim: fresh noise each time
            self._frames, self._intrinsics = self._noise_frames()
        return self._frames

    def _prepare(self, frames: dict) -> dict:
        """What the policy is fed: depth mask (range gate + optional table mask), then the noise ablation."""
        out = dict(frames)
        for slot in self.camera_slots:
            key = f"{slot}_depth"
            if key in out:
                out[key] = self.depth_mask.apply(slot, out[key], self._intrinsics.get(slot))
            if slot in self.config.noise_slots:
                if slot in out:
                    out[slot] = self._rng.integers(0, 256, out[slot].shape, dtype=np.uint8)
                if key in out:
                    out[key] = self._rng.uniform(0.0, 3.0, out[key].shape).astype(np.float32)
        return out

    def _publish_cameras(self, policy_frames: dict, source: str) -> None:
        with self._lock:
            self._cam_snap = {"raw": self._frames, "policy": policy_frames, "intrinsics": dict(self._intrinsics),
                              "t": time.time(), "source": source}

    def _maybe_preview(self) -> None:
        self._step_count += 1
        if self._step_count % max(1, int(self.control_hz // 5)) == 0:  # ~5 Hz
            src = "live cameras" if self._cam_wrapper is not None else "sim (no cameras: noise)"
            self._publish_cameras(self._prepare(self._current_frames()), f"{src}, idle preview")

    # ------------------------------------------------------------------------------------------- policy
    def _reached(self, q: np.ndarray, goal: np.ndarray) -> bool:
        if self.pool.goal_is_eef(self._active):
            T, G = self.kin.flange_pose(q), self.kin.flange_pose(goal)
            pos = float(np.linalg.norm(T[:3, 3] - G[:3, 3]))
            c = np.clip((np.trace(G[:3, :3] @ T[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0)
            return pos <= self.config.goal_tolerance_m and np.degrees(np.arccos(c)) <= self.config.goal_tolerance_deg
        return float(np.max(np.abs(goal - q))) <= self.config.goal_tolerance_rad

    def _policy_step(self) -> None:
        cfg = self.config
        with self._lock:
            goal, name = self._goal, self._active
        q = self._q
        if goal is None:
            self._set_state(State.IDLE, "no goal")
            return
        if self._reached(q, goal):
            other = None
            with self._lock:
                if cfg.alternate_goals and all(v is not None for v in self.goal_slots.values()):
                    other = "B" if self.active_slot != "B" else "A"
                    self._goal, self.active_slot = self.goal_slots[other].copy(), other
                    self.laps += 1
            if other is None:
                self._set_state(State.REACHED, f"goal reached after {self.stats.chunks} chunks, "
                                               f"{time.time() - (self.stats.started_at or time.time()):.1f} s")
                return
            logger.info("goal reached: ping-pong to %s (lap %d)", other, self.laps)
            self._chunk = None
            goal = self.goal
        if not cfg.continuous and self.stats.chunks >= cfg.max_chunks:
            self._set_state(State.IDLE, f"stopped: goal not reached within {cfg.max_chunks} chunks")
            return
        if self._cam_wrapper is not None and time.monotonic() - self._frames_seen_at > cfg.max_frame_age_s:
            self._set_state(State.IDLE, f"stopped: camera frames stale for "
                                        f"{time.monotonic() - self._frames_seen_at:.1f} s")
            return

        steps_per_wp = max(1, int(round(self.control_hz / max(cfg.exec_rate_hz, 1e-3))))
        if self._chunk is None or (self._wp_tick % steps_per_wp == 0
                                   and self._chunk_idx + 1 >= min(cfg.exec_steps, len(self._chunk))):
            policy_frames = self._prepare(self._current_frames())
            obs = make_obs(q, goal, policy_frames, self._intrinsics, step=self.stats.chunks, episode=self._episode,
                           camera_keys=self.pool[name].meta["camera_keys"],
                           depth_gate=bool(self.depth_mask.use_range_gate))
            t0 = time.perf_counter()
            try:
                chunk = chunk_of(self.pool[name].act(obs))
            except Exception as exc:  # noqa: BLE001
                logger.exception("policy call failed")
                self._set_state(State.IDLE, f"stopped: policy {name}: {type(exc).__name__}: {exc}")
                return
            self.stats.last_inference_ms = 1000.0 * (time.perf_counter() - t0)
            self.stats.last_chunk_max_step = float(np.max(np.abs(np.diff(np.vstack([q, chunk]), axis=0))))
            self.stats.chunks += 1
            self._chunk, self._chunk_idx, self._wp_tick = chunk, 0, 0
            self._publish_cameras(policy_frames, f"policy input ({name})")
            self._count_waypoint(q)
        elif self._wp_tick % steps_per_wp == 0:
            self._chunk_idx = min(self._chunk_idx + 1, len(self._chunk) - 1)
            self._count_waypoint(q)
        self._wp_tick += 1
        self._step(self._shaper(self._chunk[self._chunk_idx], q))

    def _count_waypoint(self, q: np.ndarray) -> None:
        wp = self._chunk[self._chunk_idx]
        m = self.config.joint_limit_margin
        clipped = np.clip(wp, franka.FR3_Q_MIN + m, franka.FR3_Q_MAX - m)
        if not np.allclose(clipped, wp) or float(np.max(np.abs(clipped - q))) > self.config.max_waypoint_jump:
            self.stats.clamped_waypoints += 1
        self.stats.waypoints += 1

    # --------------------------------------------------------------------------------------------- home
    def _home_step(self) -> None:
        cfg, home = self.config, self._home
        if home["t0"] is None:
            start = self._shaper.cmd if self._shaper.cmd is not None else self._q
            home.update(t0=time.monotonic(), q0=np.array(start, dtype=np.float64))
            dist = float(np.max(np.abs(home["target"] - home["q0"])))
            home["T"] = max(cfg.home_min_time, 1.875 * dist / max(cfg.home_max_speed, 1e-3))
            logger.info("homing: largest joint move %.3f rad over %.1f s", dist, home["T"])
        t = time.monotonic() - home["t0"]
        tau = min(1.0, t / home["T"])
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        target = home["q0"] + s * (home["target"] - home["q0"])
        # keep the shaper on the trajectory so a Stop decelerates from here
        prev = self._shaper.cmd
        self._shaper.cmd = target.copy()
        self._shaper.vel = (target - prev) * self.control_hz if prev is not None else np.zeros(7)
        if self._step(target) is None:
            return
        if tau >= 1.0:
            err = float(np.max(np.abs(self._q - home["target"])))
            if err <= cfg.home_settle_tol or t > home["T"] + cfg.home_settle_timeout:
                self._shaper.vel = np.zeros(7)
                self._hold = home["target"].copy()
                self._set_state(State.IDLE, f"home reached (max error {err:.3f} rad)")

    def _wind_down(self) -> None:
        """On shutdown: decelerate a motion in progress to a stop before the env is closed."""
        if self.state is State.ERROR or self._shaper.cmd is None:
            return
        self._set_state(State.IDLE, "shutting down")
        self._on_transition(State.RUNNING, State.IDLE)
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end and float(np.max(np.abs(self._shaper.vel))) > 1e-3:
            if self._step(self._shaper(self._hold, self._q)) is None:
                return
