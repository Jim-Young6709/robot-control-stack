# robots-workstation1: DRP deployment runbook

Single FR3 (FCI `192.168.1.12`) + Robotiq 2F-85 (`DAANTG8W`, `/dev/ttyUSB0`), arm key `right`.
The DRP policy runs as a vlagents server in its own env; `franka.py` (RCS venv) is the client.

```
~/drp/ckpt/<exp>/            10 ckpts of the 2026-11-19 batch (verified with ~/drp/receive_ckpts.py)
~/drp/ckpt_active/           symlinks to them
~/drp/impact_eefgoal/        Jim-Young6709/impact @ eef-goal 3757c50 (model code)
~/drp/envs/drp_vlagents/     py3.11, torch 2.8 cu128, vlagents fork (editable ~/vlagents)
~/vlagents/                  github.com/Jim-Young6709/vlagents, branch drp-agent: vlagents/policies/drp.py (agent "drp")
```

## 1. Policy server (terminal 1)

```shell
~/drp/envs/drp_vlagents/bin/python -m vlagents start-server drp --host localhost --port 20000 \
    --kwargs '{"checkpoint_path": "/home/jim/drp/ckpt/bc-rgb_pm_cam_large_clip-w150-s3-chunk1m_cam3x-rq-11132200"}' 
```

Any directory in `~/drp/ckpt/` works. `franka.json` must say `"vlagents_model": "drp"` and the same port.

## 2. Sim first (terminal 2, repo root)

```shell
# MuJoCo viewer, real time, one episode per goal in franka.json
.venv/bin/python examples/inference/franka.py --platform sim --realtime --max-steps 600 --episodes 6
# headless + numbers
.venv/bin/python examples/inference/franka.py --platform sim --headless --max-steps 600 --episodes 6
.venv/bin/python examples/inference/analyze_rollout.py --latest --plot /tmp/rollout.png
# all checkpoints x gain presets (starts its own servers)
.venv/bin/python examples/inference/sim_eval_ckpts.py
```

The sim emulates the rcs_fr3 joint impedance controller (Kp/Kd, torque limit, 1/fps target interpolation,
gravity compensated), so sim speed/tracking is indicative of the robot.

## 3. Real robot (terminal 2)

Before the first run:
- Desk: FCI active, brakes open, **end effector set to the Robotiq 2F-85 with its mass/CoM** (the torque
  controller relies on libfranka's gravity compensation of the load).
- Workspace clear, e-stop in hand. For the very first run consider `"max_joint_speed": 0.3` and a single
  nearby goal (e.g. the last one in `franka.json`, 0.35 rad from home).

```shell
.venv/bin/python examples/inference/franka.py --platform hw
```

Every arm motion, including going home, runs through the rcs_fr3 joint impedance controller
(`async_control=True`, `allow_high_collision=True`, `ignore_realtime=True` as in
`examples/fr3/fr3_env_cartesian_control.py`). Homing is a quintic trajectory streamed through that controller
(`impedance_home`, <= `home_max_speed` 0.25 rad/s) instead of rcs' stiff `move_home`.

Step-by-step first test (each motion waits for Enter, Robotiq included):

```shell
.venv/bin/python examples/inference/hw_gated_test.py --config examples/inference/franka_hw_first.json
```

`franka.py --platform hw` asks for Enter before the first homing. Keys: `e` start, `r` start + record, `s` success/reset,
`q` stop and go home, `o` reload `franka.json`, `x` exit. When an episode ends on its own (goal reached,
reflex, policy error, `--max-steps`) the robot holds its pose until you press `e` (home + next goal) or `q`.

## What `franka.py` does with a policy waypoint

1. clamp to FR3 datasheet limits minus `joint_limit_margin` (DRP was trained on Panda limits),
2. scale the jump from the measured joints down to `max_waypoint_jump` (0.35 rad),
3. rate-limit the commanded target to `max_joint_speed` / `max_joint_accel` (0.5 rad/s, 2 rad/s^2),
4. rcs LimitedAbsoluteAction (`max_rel_mov_joints` 0.35, per joint) as backstop, then the joint impedance
   controller (`async_control=True`, `kp`/`kd`/`torque_limit` from `franka.json`).

## Gains

`franka.json` uses Kp=[600,600,600,600,250,150,50], Kd=[50,50,50,50,30,25,15] with the rcs default 5 Nm torque
limit. The rcs default Kp=100/Kd=20 is too soft for these checkpoints: their chunk[0] is only 0.02-0.04 rad
ahead of the measured joints, which at Kp=100 is ~2-4 Nm against ~1.1 Nm joint friction, so the arm crawls
(0/6 goals in 20 s in sim). Safety comes from the 5 Nm per joint torque cap either way.

## Cameras

`camera_source: "noise"` = blind test (uniform RGB 0-255, depth 0-3 m, nominal D415 intrinsics). This only
validates the pipeline: DRP was trained on two fixed cameras `front_left` / `front_right` behind the base
(nominal eye (-0.35, +-1.0, 1.2) m looking at (0.5, -+0.05, 0.5), 320x240, D415 FOV), not on a wrist camera.

## Real-world eval: the DRP viser panel on RCS

`drp_viser_app.py` is the deployment UI: the controls of the previous robot PC's panel (checkpoint dropdown,
activate / stop / closed-loop rollout, exec sliders, EE gizmo + IK goal, saved goals A/B + ping-pong, home + recover,
gripper, camera views with noise ablation and table mask, 3D scene), built on RCS:

- robot, gripper and cameras through the RCS env (`drp_viser/envs.py`): every motion, homing included, through
  rcs_fr3's joint impedance controller (gains/limits from `franka.json`), the Robotiq through rcs_robotiq2f85 with
  activation on the panel's button (`activate_on_init=False`), the two front D415s through rcs_realsense with
  `DRP_D415_KWARGS` (320x240 colour, 848x480 depth aligned onto it, the drp_deploy depth post-processing
  disparity -> spatial -> temporal -> depth -> hole filling, true intrinsics);
- one vlagents `drp` server per checkpoint of `~/drp/ckpt_active` on ports 20101.. (`drp_viser/policies.py`), all
  started at launch, so switching checkpoints is instant; the agent's meta query tells the panel each checkpoint's
  inputs;
- `drp_viser/runner.py`: one thread owns the env and steps it at 30 Hz (hold / policy waypoints through
  `WaypointShaper` / quintic homing); the UI only changes state and settings;
- FR3 kinematics (gizmo IK, eef goal check) and the 3D scene from RCS's FR3 / Robotiq MJCF models.

Depth pipeline, every step switchable live in the panel's "Depth processing" tab (left panel):

1. camera (rcs_realsense, per frame): IR emitter; 848x480 depth aligned onto the 320x240 colour (fixed); then
   disparity domain -> spatial filter -> temporal filter -> back to depth -> hole filling (drp_deploy chain,
   librealsense defaults); a device that stops delivering frames is restarted (`restart_attempts` 3);
2. panel: range gate 0.01-3 m (the renders' clipping range; OFF also tells the policy server to skip its gate),
   optional table-plane mask (fit, margin), per-camera noise ablation (Cameras tab);
3. policy server (vlagents drp): metres x depth_scale (2 mm steps), pointmap from the true K.

Measured on the two D415s (no depth / in 0.01-3 m): defaults 0.1-0.8% / 77-93%; hole filling off 6-8% / 73-77%;
also temporal off 9-11%; raw aligned depth 9-11% / 70-76%; emitter off as well 9-15%.

```shell
~/launch_scripts/realworld_launch.sh                # = examples/inference/realworld_launch.sh: checks, then the panel
~/launch_scripts/realworld_launch.sh --blind        # robot without cameras (noise views)
~/launch_scripts/realworld_launch.sh --platform sim # RCS MuJoCo sim dry run
~/launch_scripts/realworld_launch.sh --preflight-only
```

The launcher runs `realworld_preflight.py` (read-only: robot mode, Desk end effector mass, other FCI clients /
panels / camera users, both D415s on USB3, the Robotiq adapter, GPU, checkpoints, the fork patches), starts the panel,
opens it in the browser and keeps everything in `~/drp/logs/realworld/<date>-<time>/` (session log + policy server
logs). Nothing moves until a button is pressed: once connected the impedance controller holds the measured pose.
Ctrl-C once: the arm is brought to rest, robot / cameras / gripper are closed, then the policy servers are stopped.

`franka.py` stays the blind (noise) client of a single policy server.
