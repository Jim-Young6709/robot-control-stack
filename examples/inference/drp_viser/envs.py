"""RCS envs for the panel, built like examples/inference/franka.py builds them, plus what the panel needs:

* the two front RealSense cameras through rcs_realsense with the DRP capture (CAMERA_RIGS: the D455 pair at 640x480,
  or the D415 pair at 320x240; depth aligned onto colour, drp_deploy depth post-processing, true intrinsics),
* a continuous gripper action (closure slider) and Robotiq activation on demand (activate_on_init=False), so creating
  the env and resetting it moves neither the arm nor the gripper.

The env must be created, reset, stepped and closed in one thread (RCS sim binds greenlets to the creating thread).
"""

import gymnasium as gym
import numpy as np
from rcs._core.sim import SimConfig

import franka  # examples/inference/franka.py: station constants, InferenceConfig, JointImpedanceSimEmulation

import rcs

#: Front camera pairs of the DRP cell. Fixed binding: left/right is the ROBOT's (both stand behind the base); in
#: front_left the arm is on the right of the image, in front_right on the left. Select with --camera-rig.
CAMERA_RIGS = {
    # since 2026-11-25: two D455 at 640x480 colour + depth (79.4 x 63.9 deg, fx ~385); the panel downscales 2x to the
    # policy's 320x240. Binding checked from images: 213522250587 has the arm on the right of its image.
    "d455": {"serials": {"front_left": "213522250587", "front_right": "234222300688"},
             "size": (640, 480), "kwargs": "DRP_D455_KWARGS"},
    # until 2026-11-25: two D415 at 320x240 colour + 848x480 depth (54.8 x 42.5 deg), kept for switching back
    "d415": {"serials": {"front_left": "747612060067", "front_right": "947122060531"},
             "size": (320, 240), "kwargs": "DRP_D415_KWARGS"},
}
DEFAULT_CAMERA_RIG = "d455"
FRONT_D415 = CAMERA_RIGS["d415"]["serials"]
CAMERA_FPS = 30


def make_hw_env(cfg: "franka.InferenceConfig", cameras: bool = True, camera_fps: int = CAMERA_FPS,
                camera_rig: str = DEFAULT_CAMERA_RIG) -> gym.Env:
    """cfg.fps is the control rate: env step pacing and the impedance controller's interpolation window."""
    from rcs_fr3.configs import DefaultFR3HardwareEnv
    from rcs_fr3.creators import (
        FR3MultiHardwareEnvCreatorConfig,
        HardwareCameraCreatorConfig,
        RCSFR3MultiConfigEnvCreator,
    )
    from rcs_robotiq2f85.hw import RobotiQ2F85GripperConfig

    base = DefaultFR3HardwareEnv()
    base.ip = franka.ROBOT_IP
    robot_cfg = base.config().robot_cfg
    # torque level joint impedance controller (Franka::joint_controller) fed by streamed joint targets
    robot_cfg.async_control = True
    robot_cfg.kp = np.asarray(cfg.kp, dtype=np.float64)
    robot_cfg.kd = np.asarray(cfg.kd, dtype=np.float64)
    robot_cfg.torque_limit = np.asarray(cfg.torque_limit, dtype=np.float64)
    # 100 Nm / 100 N collision thresholds while the controller runs (examples/fr3/fr3_env_cartesian_control.py)
    robot_cfg.allow_high_collision = cfg.allow_high_collision
    # on (re)start of the controller, ramp from the measured pose to the first target
    robot_cfg.blocking_move_on_start = True
    # the controller interpolates each target over 1/policy_rate: must equal the step rate
    robot_cfg.policy_rate = int(cfg.fps)
    robot_cfg.ignore_realtime = True
    robot_cfg.speed_factor = 0.1
    robot_cfg.tcp_offset = rcs.GRIPPER_TCP_OFFSETS[rcs.common.GripperType("Robotiq2F85")]
    robot_cfg.q_home = np.asarray(franka.Q_HOME_DEFAULT, dtype=np.float64)

    camera_cfgs = None
    if cameras:
        import rcs_realsense.camera

        rig = CAMERA_RIGS[camera_rig]
        width, height = rig["size"]
        camera_cfgs = {
            "front": HardwareCameraCreatorConfig(
                camera_type_id="realsense",
                camera_cfgs={
                    slot: rcs.common.BaseCameraConfig(identifier=serial, resolution_width=width,
                                                      resolution_height=height, frame_rate=int(camera_fps))
                    for slot, serial in rig["serials"].items()
                },
                kwargs=dict(getattr(rcs_realsense.camera, rig["kwargs"])),
            )
        }
    hw_cfg = FR3MultiHardwareEnvCreatorConfig(
        robot_cfgs={franka.ROBOT_KEY: robot_cfg},
        control_mode=franka.CONTROL_MODE,
        gripper_cfgs={franka.ROBOT_KEY: RobotiQ2F85GripperConfig(
            serial_number=franka.ROBOTIQ_SERIAL, speed=150, force=100, async_control=True, activate_on_init=False)},
        camera_cfgs=camera_cfgs,
        max_relative_movement=cfg.max_rel_mov_joints,
        relative_to=franka.RELATIVETO,
        robot_to_shared_base_frame={franka.ROBOT_KEY: rcs.common.Pose()},
        frequency=cfg.fps,
    )
    hw_cfg.wrapper_cfg.binary_gripper = False
    hw_cfg.wrapper_cfg.include_depth = cameras
    hw_cfg.wrapper_cfg.home_on_reset = False  # homing is the panel's impedance trajectory
    return RCSFR3MultiConfigEnvCreator().create_env(hw_cfg)


def make_sim_env(cfg: "franka.InferenceConfig", headless: bool = True) -> gym.Env:
    """FR3 + Robotiq in MuJoCo with the emulated joint impedance controller, real time, no cameras (the panel feeds
    noise frames in sim)."""
    from rcs.envs.configs import EmptyWorldDroid

    scene = EmptyWorldDroid()
    sim_cfg = scene.config()
    sim_cfg.sim_cfg = SimConfig(async_control=True, realtime=True, frequency=cfg.fps, max_convergence_steps=500)
    sim_cfg.camera_cfgs = None
    sim_cfg.camera_adds = None
    sim_cfg.robot_frame_objects = None
    sim_cfg.control_mode = franka.CONTROL_MODE
    sim_cfg.relative_to = franka.RELATIVETO
    sim_cfg.max_relative_movement = cfg.max_rel_mov_joints
    sim_cfg.wrapper_cfg.binary_gripper = False
    sim_cfg.wrapper_cfg.include_depth = False
    sim_cfg.headless = headless
    robot_cfg = sim_cfg.robot_cfgs[franka.ROBOT_KEY]
    robot_cfg.q_home = np.asarray(franka.Q_HOME_DEFAULT, dtype=np.float64)
    robot_cfg.kp = list(cfg.kp)
    robot_cfg.kv = list(cfg.kd)
    env = scene.create_env(sim_cfg)
    return franka.JointImpedanceSimEmulation(env, franka.ROBOT_KEY, cfg.torque_limit)
