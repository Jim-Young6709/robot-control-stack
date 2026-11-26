#!/usr/bin/env python3
"""DRP viser panel on robot-control-stack (robots-workstation1: FR3 + Robotiq 2F-85 + two front RealSense cameras).

The controls of the previous robot PC's panel (checkpoint dropdown, activate / stop / closed-loop rollout, exec
sliders, EE gizmo + IK goal, saved goals A/B + ping-pong, home + recover, gripper, camera views with noise ablation and
table mask, 3D scene), built on RCS:

* robot, gripper and cameras through the RCS env (drp_viser/envs.py): rcs_fr3 joint impedance controller for every
  motion (homing included), rcs_robotiq2f85, rcs_realsense with the DRP depth pipeline;
* every checkpoint of ~/drp/ckpt_active as its own vlagents `drp` server (drp_viser/policies.py), all started at
  launch, so switching checkpoints is instant;
* FR3 kinematics (gizmo IK) and the 3D scene from RCS's FR3 / Robotiq models.

    .venv/bin/python examples/inference/drp_viser_app.py                  # the real eval: robot, cameras, gripper
    .venv/bin/python examples/inference/drp_viser_app.py --blind          # robot without cameras: noise views
    .venv/bin/python examples/inference/drp_viser_app.py --platform sim   # RCS MuJoCo sim, noise views

Usually started by examples/inference/realworld_launch.sh (pre-flight checks + session log). Nothing moves until a
button is pressed: once connected, the joint impedance controller holds the measured pose.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import franka  # noqa: E402
from drp_viser.depth import DepthMask  # noqa: E402
from drp_viser.envs import CAMERA_RIGS, DEFAULT_CAMERA_RIG  # noqa: E402
from drp_viser.policies import BASE_PORT, CKPT_ROOT, PolicyPool  # noqa: E402
from drp_viser.runner import PanelRunner, RunConfig  # noqa: E402
from rcs._core.common import RobotPlatform  # noqa: E402

logger = logging.getLogger("drp_viser")

#: #1 of the 2026-11-19 checkpoint set (sim hard 0.727, ood5x 0.652)
DEFAULT_CHECKPOINT = "bc-rgb_pm_cam_large_clip-w150-s3-chunk1m_cam3x-rq-11132200"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--platform", choices=["hw", "sim"], default="hw")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080, help="viser port; the next free one is used if taken")
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT), help="checkpoint directories (+ ckpt_info.json)")
    p.add_argument("--checkpoint", default="", help=f"checkpoint to start on (default {DEFAULT_CHECKPOINT})")
    p.add_argument("--policy-port", type=int, default=BASE_PORT, help="port of the first policy server")
    p.add_argument("--log-dir", default=str(Path.home() / "drp/logs/realworld"), help="policy server logs")
    p.add_argument("--blind", action="store_true", help="hardware without opening the cameras: noise in both views")
    p.add_argument("--control-hz", type=int, default=70,
                   help="control loop rate: env steps, impedance interpolation window, upper limit of the policy rate")
    p.add_argument("--policy-hz", type=float, default=30.0, help="initial policy (exec) rate, the GUI slider changes it")
    p.add_argument("--camera-fps", type=int, choices=[30, 60], default=30, help="camera frame rate (colour and depth)")
    p.add_argument("--camera-rig", choices=sorted(CAMERA_RIGS), default=DEFAULT_CAMERA_RIG,
                   help="front camera pair: d455 (640x480, 79 deg, current) or d415 (320x240, 55 deg, previous)")
    p.add_argument("--camera-crop", type=float, default=1.0,
                   help="centre crop before resizing to the policy's 320x240 (0.625 on the D455 = the D415's field)")
    p.add_argument("--noise", nargs="*", default=None, metavar="SLOT",
                   help="camera slots to start with 'Noise out' ON (front_left, front_right)")
    p.add_argument("--config", default=None, help="franka.json-style config (gains, limits, fps); default franka.json")
    p.add_argument("--kp", type=float, nargs=7, default=None, help="joint impedance stiffness (default: config)")
    p.add_argument("--kd", type=float, nargs=7, default=None, help="joint impedance damping (default: config)")
    p.add_argument("--torque-limit", type=float, nargs=7, default=None, help="impedance torque cap [Nm]")
    p.add_argument("--no-high-collision", action="store_true", help="keep libfranka's 20/10 Nm collision thresholds")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    if args.config:
        franka.CONFIG_PATH = Path(args.config)
    cfg = franka.load_inference_config()
    for key in ("kp", "kd", "torque_limit"):
        if getattr(args, key) is not None:
            setattr(cfg, key, list(getattr(args, key)))
    if args.no_high_collision:
        cfg.allow_high_collision = False
    platform = RobotPlatform.HARDWARE if args.platform == "hw" else RobotPlatform.SIMULATION
    cfg.fps = int(args.control_hz)  # the panel's control rate (franka.json's fps stays franka.py's)
    run_cfg = RunConfig(
        fps=float(cfg.fps), max_waypoint_jump=cfg.max_waypoint_jump, max_joint_speed=cfg.max_joint_speed,
        max_joint_accel=cfg.max_joint_accel, joint_limit_margin=cfg.joint_limit_margin,
        goal_tolerance_rad=cfg.goal_tolerance_rad, exec_rate_hz=min(float(args.policy_hz), float(cfg.fps)),
        home_max_speed=cfg.home_max_speed,
        home_min_time=cfg.home_min_time, home_settle_tol=cfg.home_settle_tol,
        home_settle_timeout=cfg.home_settle_timeout,
    )
    slots = ["front_left", "front_right"]
    noise = list(slots) if (args.blind or platform == RobotPlatform.SIMULATION) else (args.noise or [])
    unknown = sorted(set(noise) - set(slots))
    if unknown:
        print(f"--noise: unknown camera slot(s) {unknown}; have {slots}", file=sys.stderr)
        return 2
    run_cfg.noise_slots = set(noise)
    if not 0.2 <= args.camera_crop <= 1.0:
        print(f"--camera-crop {args.camera_crop}: use 0.2-1.0", file=sys.stderr)
        return 2
    run_cfg.camera_crop = float(args.camera_crop)

    # ------------------------------------------------------------------ policy servers (they take a while: first)
    pool = PolicyPool.from_dir(Path(args.ckpt_root).expanduser(), args.policy_port, Path(args.log_dir).expanduser())
    if not pool.names:
        print(f"no checkpoints (model_config.yaml + latest.pt) under {args.ckpt_root}", file=sys.stderr)
        return 2
    active = args.checkpoint or (DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT in pool.names else pool.names[0])
    if active not in pool.names:  # a typo must not silently run another checkpoint on the arm
        print(f"--checkpoint {active!r} not found; have: {', '.join(pool.names)}", file=sys.stderr)
        return 2
    print(f"checkpoints ({len(pool.names)}), starting on {active}:")
    for name in pool.names:
        print(f"  port {pool[name].port}  {pool.label(name)}")
    pool.start_background(first=active)

    runner = server = stop_ui = None

    def cleanup() -> None:
        # arm first (decelerate, close the env: FCI, cameras, gripper port), then the UI, then the policy servers
        if stop_ui is not None:
            stop_ui.set()
        if runner is not None:
            runner.shutdown()
        if server is not None:
            try:
                server.stop()
            except Exception:  # noqa: BLE001
                logger.exception("viser stop failed")
        pool.stop_all()

    def _raise_interrupt(signum, frame):
        raise KeyboardInterrupt

    # kill and a closed terminal take the same path as Ctrl-C, so the robot, cameras and servers are released
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)

    try:
        from drp_viser.envs import make_hw_env, make_sim_env
        from drp_viser.kinematics import RcsKinematics

        m = cfg.joint_limit_margin
        kin = RcsKinematics(joint_limits=np.stack([franka.FR3_Q_MIN + m, franka.FR3_Q_MAX - m]))
        if platform == RobotPlatform.HARDWARE:
            print(f"connecting to the FR3 (control {cfg.fps} Hz), the gripper port"
                  + ("" if args.blind else f" and the front {args.camera_rig.upper()}s "
                     f"({'x'.join(map(str, CAMERA_RIGS[args.camera_rig]['size']))} @ {args.camera_fps} fps, "
                     f"crop {args.camera_crop:g} -> 320x240)") + " ...")
            make_env = lambda: make_hw_env(cfg, cameras=not args.blind, camera_fps=args.camera_fps,  # noqa: E731
                                           camera_rig=args.camera_rig)
        else:
            print("starting the RCS MuJoCo sim (FR3 + Robotiq, emulated joint impedance, noise cameras) ...")
            make_env = lambda: make_sim_env(cfg, headless=True)  # noqa: E731
        runner = PanelRunner(make_env, platform, pool, active, kin, run_cfg, DepthMask(), slots)
        runner.start()
        if not runner.ready.wait(timeout=150.0):
            raise RuntimeError("the RCS env did not come up within 150 s")
        if runner.startup_error:
            raise RuntimeError(runner.startup_error)
        snap = runner.snapshot()
        print(f"robot: mode {snap['robot_mode']}, holding q = {np.round(snap['q'], 3).tolist()}")

        import viser

        from drp_viser.gui import build_gui
        from drp_viser.scene import RcsRobotScene

        server = viser.ViserServer(host=args.host, port=args.port, label="DRP / FR3 (RCS)")
        scene = RcsRobotScene(server)
        stop_ui = build_gui(server, runner, pool, scene, kin, active)
    except KeyboardInterrupt:
        print("\ninterrupted during startup, releasing everything ...", file=sys.stderr)
        cleanup()
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("startup failed")
        print(f"\nstartup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        cleanup()
        print("released everything that had been opened.", file=sys.stderr)
        return 2

    # one write, so that log lines of other threads cannot land inside it (the launcher parses this line)
    sys.stdout.write(f"\n  UI: http://localhost:{server.get_port()}\n"
                     "  Nothing moves until you press a button. Ctrl-C to quit.\n\n")
    sys.stdout.flush()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nshutting down: stopping the arm, closing the robot, cameras, gripper and policy servers ...")
    finally:
        cleanup()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
