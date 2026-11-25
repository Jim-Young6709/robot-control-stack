"""Gated first hardware test on robots-workstation1 (FR3 + Robotiq), every motion confirmed by a human.

Before each motion the script prints `READY_FOR_STEP_<n>: <what will move>` and waits for confirmation:
  interactive (default): press Enter to run the step, 'a' + Enter to abort; Ctrl-C aborts during a motion.
  --gate-dir DIR: waits until the file DIR/go_<n> exists; DIR/abort stops at any gate or control step.
The e-stop is the primary safety, this only makes sure nothing moves unannounced.

Steps:
  0  connect + Robotiq activation (the fingers move during activation)      [skipped with --no-gripper]
  1  move home with the joint impedance controller (quintic, <= home_max_speed), gripper opens
  2  joint impedance check: joint 7 does one smooth 0 -> +amp -> 0 bump, all other joints hold
  3  gripper check: close, hold 2 s, open (arm holds its pose)               [skipped with --no-gripper]
  4  DRP episode to the (first) goal of the config, noise RGB-D, waypoint shaping as in franka.py
  5  move home with the joint impedance controller
Every arm motion goes through rcs_fr3's torque level joint impedance controller; it keeps running (holding the
pose compliantly) from step 1 until the script exits.

    .venv/bin/python examples/inference/hw_gated_test.py --config examples/inference/franka_hw_first.json
"""

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import franka  # noqa: E402
from rcs._core.common import RobotPlatform  # noqa: E402
from rcs_fr3._core import hw  # noqa: E402

HW = RobotPlatform.HARDWARE


def say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Gate:
    def __init__(self, gate_dir: Path | None):
        self.dir = gate_dir
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            for f in self.dir.glob("*"):
                f.unlink()

    def aborted(self) -> bool:
        return self.dir is not None and (self.dir / "abort").exists()

    def wait(self, n: int, desc: str) -> bool:
        say(f"READY_FOR_STEP_{n}: {desc}")
        if self.dir is None:
            try:
                answer = input(f"    >>> step {n}: press Enter to EXECUTE (e-stop in hand), 'a' + Enter to abort: ")
            except EOFError:
                answer = "a"
            if answer.strip().lower().startswith("a"):
                say(f"ABORT before step {n}")
                return False
            say(f"GO_{n}")
            return True
        while True:
            if self.aborted():
                say(f"ABORT before step {n}")
                return False
            if (self.dir / f"go_{n}").exists():
                say(f"GO_{n}")
                return True
            time.sleep(0.1)


def make_action(q: np.ndarray, gripper: float | None, use_gripper: bool) -> dict:
    act = {"joints": np.asarray(q, dtype=np.float64)}
    if use_gripper:
        act["gripper"] = np.asarray([1.0 if gripper is None else gripper], dtype=np.float32)
    return {franka.ROBOT_KEY: act}


def gripper_width(obs) -> float:
    return float(np.asarray(obs[franka.ROBOT_KEY].get("gripper", [np.nan])).reshape(-1)[0])


def joints(obs) -> np.ndarray:
    return np.asarray(obs[franka.ROBOT_KEY]["joints"], dtype=np.float64)


def robot_state(obs) -> dict:
    return obs[franka.ROBOT_KEY].get("robot_state", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("franka_hw_first.json")))
    parser.add_argument("--gate-dir", default=None, help="file based gates instead of Enter prompts")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--j7-amp", type=float, default=0.15, help="joint 7 bump amplitude [rad]")
    parser.add_argument("--j7-period", type=float, default=4.0, help="joint 7 bump duration [s]")
    parser.add_argument("--no-gripper", action="store_true", help="arm only, do not connect the Robotiq")
    args = parser.parse_args()

    franka.CONFIG_PATH = Path(args.config).expanduser().resolve()
    cfg = dataclasses.replace(franka.load_inference_config(), use_gripper=not args.no_gripper)
    goal = franka.goal_for_episode(cfg, 0)
    q_home = np.asarray(cfg.q_home if cfg.q_home is not None else franka.Q_HOME_DEFAULT, dtype=np.float64)
    gate = Gate(Path(args.gate_dir) if args.gate_dir else None)
    out_dir = Path(cfg.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    say(f"config {franka.CONFIG_PATH}: kp={cfg.kp} kd={cfg.kd} torque_limit={cfg.torque_limit} "
        f"max_joint_speed={cfg.max_joint_speed} fps={cfg.fps} goal={None if goal is None else goal.tolist()}")

    if cfg.use_gripper and not gate.wait(
        0, f"connect the arm (no motion) and ACTIVATE the Robotiq {franka.ROBOTIQ_SERIAL}: fingers move during activation"
    ):
        return
    env = franka.get_hw_env(cfg)  # connects to the FCI (no arm motion) and activates the Robotiq
    robot = env.get_wrapper_attr("robot")[franka.ROBOT_KEY]
    st = robot.get_state().robot_state
    q_now = np.asarray(st.q)
    say(f"connected{' + Robotiq activated' if cfg.use_gripper else ''}. robot_mode={st.robot_mode} q={np.round(q_now, 3).tolist()} "
        f"max|q-home|={np.abs(q_now - q_home).max():.3f} rad m_ee={st.m_ee:.2f} kg")
    if st.robot_mode not in (hw.RobotMode.kIdle, hw.RobotMode.kMove):
        say(f"NOT_READY: robot_mode={st.robot_mode}, unlock the joints and activate FCI in Desk")
        env.close()
        return
    ctrl = franka.ModelInference(env, cfg, HW, max_steps=args.max_steps)  # connects to the policy server
    say(f"policy server '{cfg.vlagents_model}' on port {cfg.vlagents_port} ready")
    obs = None
    try:
        # ------------------------------------------------------------------ 1: home
        if not gate.wait(1, f"move HOME {np.round(q_home, 3).tolist()} with the JOINT IMPEDANCE controller "
                            f"(<= {cfg.home_max_speed} rad/s, largest joint move {np.abs(q_now - q_home).max():.2f} rad)"
                            f"{', gripper opens' if cfg.use_gripper else ''}"):
            return
        obs, _ = env.reset()  # error recovery + gripper open, no arm motion (home_on_reset=False)
        obs = franka.impedance_home(env, obs, cfg, q_home, abort_check=gate.aborted)
        say(f"STEP_1_DONE home reached, |q-home|max={np.abs(joints(obs) - q_home).max():.4f} rad, "
            f"gripper width={gripper_width(obs):.2f}")

        # ------------------------------------------------------------------ 2: joint impedance check on j7
        if not gate.wait(2, f"joint impedance check: joint 7 smooth bump 0 -> +{args.j7_amp} -> 0 rad over "
                            f"{args.j7_period:.0f} s, other joints hold (Kp/Kd/torque from config)"):
            return
        q0 = joints(obs)
        n = int(round(args.j7_period * cfg.fps))
        rec = {k: [] for k in ("q", "target", "dq", "tau_J", "tau_ext")}
        status = "completed"
        for k in range(n + int(cfg.fps)):  # bump plus 1 s hold
            if gate.aborted():
                status = "aborted"
                break
            t = min(k / cfg.fps, args.j7_period)
            tgt = q0.copy()
            tgt[6] += args.j7_amp * (1 - np.cos(2 * np.pi * t / args.j7_period)) / 2
            obs, _, term, trunc, info = env.step(make_action(tgt, 1.0, cfg.use_gripper))
            if term or trunc:
                status = f"truncated (reflex/error) info={info}"
                break
            rs = robot_state(obs)
            rec["q"].append(joints(obs))
            rec["target"].append(tgt)
            rec["dq"].append(rs.get("dq", np.full(7, np.nan)))
            rec["tau_J"].append(rs.get("tau_J", np.full(7, np.nan)))
            rec["tau_ext"].append(rs.get("tau_ext_hat_filtered", np.full(7, np.nan)))
        arr = {k: np.asarray(v) for k, v in rec.items() if v}
        if arr:
            np.savez(out_dir / f"{stamp}_hw_j7check.npz", **arr)
            err = arr["target"] - arr["q"]
            say(f"STEP_2_DONE ({status}) j7 tracking: max|err|={np.abs(err[:, 6]).max():.4f} rad, "
                f"final err={err[-1, 6]:+.4f}, max|dq7|={np.nanmax(np.abs(arr['dq'][:, 6])):.3f} rad/s | "
                f"other joints max drift={np.abs(arr['q'][:, :6] - q0[:6]).max():.4f} rad | "
                f"max|tau_ext|={np.nanmax(np.abs(arr['tau_ext'])):.2f} Nm")
        else:
            say(f"STEP_2_DONE ({status}) no data")
        if status != "completed":
            return

        # ------------------------------------------------------------------ 3: gripper check
        if cfg.use_gripper:
            if not gate.wait(3, "GRIPPER: close (no object), hold 2 s, open again; the arm holds its pose"):
                return
            q_hold = joints(obs)
            widths = []
            for cmd in [0.0] * int(2 * cfg.fps) + [1.0] * int(2 * cfg.fps):
                if gate.aborted():
                    break
                obs, _, term, trunc, info = env.step(make_action(q_hold, cmd, True))
                if term or trunc:
                    say(f"truncated (reflex/error) info={info}")
                    return
                widths.append(gripper_width(obs))
            w = np.asarray(widths)
            say(f"STEP_3_DONE gripper width: min {w.min():.2f} (closed) end {w[-1]:.2f} (open), "
                f"arm drift {np.abs(joints(obs) - q_hold).max():.4f} rad")

        # ------------------------------------------------------------------ 4: DRP episode
        if goal is None:
            say("no goal in config, skipping step 4")
        elif gate.wait(4, f"DRP episode -> goal {np.round(goal, 3).tolist()}, noise RGB-D, max {args.max_steps} "
                          f"steps ({args.max_steps / cfg.fps:.0f} s), speed cap {cfg.max_joint_speed} rad/s"):
            ctrl.goal = goal
            ctrl.episode = 0
            ctrl.step_in_episode = 0
            ctrl.shaper.reset()
            ctrl.log.start({"episode": 0, "goal_joints": goal.tolist(), "model": cfg.vlagents_model,
                            "fps": cfg.fps, "max_joint_speed": cfg.max_joint_speed, "kp": cfg.kp, "kd": cfg.kd,
                            "torque_limit": cfg.torque_limit, "camera_source": cfg.camera_source,
                            "max_rel_mov_joints": cfg.max_rel_mov_joints})
            obs_dict = ctrl.obs_rcs2agents(obs)
            reason = f"max_steps={args.max_steps}"
            for k in range(args.max_steps):
                if gate.aborted():
                    reason = "aborted"
                    break
                t0 = time.perf_counter()
                action = ctrl.act(copy.deepcopy(obs_dict))
                a = ctrl.action_agents2rcs(action, obs)
                latency = time.perf_counter() - t0
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    reason = f"truncated (reflex/error) info={info}"
                    break
                rs = robot_state(obs)
                ctrl.log.add(t=time.time(), q=joints(obs), dq=rs.get("dq"),
                             target=info[franka.ROBOT_KEY].get("limited_absolute_action", a[franka.ROBOT_KEY]["joints"]),
                             policy_action=action[franka.ROBOT_KEY].action,
                             gripper_cmd=a[franka.ROBOT_KEY].get("gripper", [np.nan]), gripper=[gripper_width(obs)],
                             latency=[latency], collision=[0.0])
                ctrl.step_in_episode += 1
                obs_dict = ctrl.obs_rcs2agents(obs)
                err = np.abs(joints(obs) - goal).max()
                if k % 15 == 0:
                    say(f"  step {k:3d}: max|q-goal|={err:.3f} rad  max|dq|={np.max(np.abs(rs.get('dq', [0]))):.3f}")
                if err <= cfg.goal_tolerance_rad:
                    reason = f"goal reached (max joint error {err:.3f} rad)"
                    break
            path = ctrl.log.save("ep000", reason)
            say(f"STEP_4_DONE {reason} after {ctrl.step_in_episode} steps, log {path}")
        else:
            return

        # ------------------------------------------------------------------ 5: home
        if not gate.wait(5, f"move HOME again with the JOINT IMPEDANCE controller "
                            f"(<= {cfg.home_max_speed} rad/s, largest joint move "
                            f"{np.abs(joints(obs) - q_home).max():.2f} rad)"):
            return
        obs, _ = env.reset()
        obs = franka.impedance_home(env, obs, cfg, q_home, abort_check=gate.aborted)
        say(f"STEP_5_DONE home reached, |q-home|max={np.abs(joints(obs) - q_home).max():.4f} rad")
    except KeyboardInterrupt:
        say("ABORTED by Ctrl-C, stopping the controller (the robot holds its pose)")
        ctrl.log.save("ep000", "aborted by ctrl-c")
    except Exception as e:  # noqa: BLE001
        say(f"ERROR {type(e).__name__}: {e}")
        raise
    finally:
        env.close()  # stops the joint impedance thread, from here on libfranka's idle hold keeps the pose
        ctrl.remote_agent.close()
        say("FINISHED, controller stopped")


if __name__ == "__main__":
    main()
