"""viser 3D scene of the FR3 + Robotiq 2F-85 from the RCS MJCF assets (RCS ships no URDF, so ViserUrdf cannot be used).

The model is composed like the RCS sim scenes (rcs.sim.composer.ModelComposer: FR3 MJCF plus the Robotiq MJCF at the
attachment site with rcs.GRIPPER_MOUNT_OFFSETS), compiled with MuJoCo only for kinematics (no Sim, no renderer) and
drawn as one viser frame per body holding that body's meshes. update() runs mj_kinematics and moves the body frames.
"""

import warnings

import mujoco
import numpy as np
from rcs.sim.composer import ModelComposer

import rcs
from rcs import common

ROBOTIQ = common.GripperType("Robotiq2F85")
FINGER_CLOSED_RAD = 0.824  # 2F-85 MJCF hinge angle at actuator ctrl 255 (fully closed), from a MuJoCo rollout


def _compose(gripper: bool) -> mujoco.MjModel:
    comp = ModelComposer(model_name="rcs_viser")
    comp.add_robot(rcs.ROBOTS[common.RobotType.FR3].mjcf_model_path, prefix="")
    if gripper:
        comp.add_gripper(rcs.GRIPPER_PATHS[ROBOTIQ], robot_prefix="", gripper_prefix="gripper_",
                         attachment_site_name="attachment_site", pose=rcs.GRIPPER_MOUNT_OFFSETS[ROBOTIQ])
    with warnings.catch_warnings():  # "Attach conflict" about solver options, irrelevant here
        warnings.simplefilter("ignore")
        return comp.spec.compile()


def _wxyz(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


class _Body:
    """One copy of the robot in the viser scene."""

    def __init__(self, server, root: str, gripper: bool, collision_meshes: bool, color=None, opacity=None):
        self.server = server
        self.m = _compose(gripper)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"fr3_joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = np.array([m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith("gripper_")],
                                    dtype=np.int64)
        self.site = m.site("attachment_site").id
        self.gripper_base = m.body("gripper_base").id if gripper else -1
        self.gripper_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("gripper_")}
        group = 3 if collision_meshes else 2  # MJCF geom groups: 2 visual, 3 collision
        self.root = server.scene.add_frame(root, show_axes=False)
        self.frames: dict[int, object] = {}
        for g in range(m.ngeom):
            if m.geom_group[g] != group or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            b = int(m.geom_bodyid[g])
            name = f"{root}/{m.body(b).name}"
            if b not in self.frames:
                self.frames[b] = server.scene.add_frame(name, show_axes=False)
            mid = int(m.geom_dataid[g])
            v0, nv, f0, nf = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            server.scene.add_mesh_simple(
                f"{name}/geom{g}", np.asarray(m.mesh_vert[v0:v0 + nv], dtype=np.float32),
                np.asarray(m.mesh_face[f0:f0 + nf], dtype=np.uint32),
                color=tuple(float(c) for c in (color if color is not None else rgba[:3])), opacity=opacity,
                position=tuple(m.geom_pos[g]), wxyz=tuple(m.geom_quat[g]))

    def update(self, q, closure, coupler: np.ndarray | None) -> None:
        self.d.qpos[self.arm_qadr] = np.asarray(q, dtype=np.float64).reshape(7)
        if closure is not None and self.finger_qadr.size:
            self.d.qpos[self.finger_qadr] = FINGER_CLOSED_RAD * float(np.clip(closure, 0.0, 1.0))
        mujoco.mj_kinematics(self.m, self.d)
        fix = None
        if coupler is not None and self.gripper_base >= 0:
            T_f = np.eye(4)
            T_f[:3, :3], T_f[:3, 3] = self.d.site_xmat[self.site].reshape(3, 3), self.d.site_xpos[self.site]
            T_b = np.eye(4)
            T_b[:3, :3], T_b[:3, 3] = self.d.xmat[self.gripper_base].reshape(3, 3), self.d.xpos[self.gripper_base]
            fix = T_f @ coupler @ np.linalg.inv(T_b)  # moves the mounted gripper from the RCS mount to the coupler pose
        with self.server.atomic():
            for b, h in self.frames.items():
                p, R = self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()
                if fix is not None and b in self.gripper_bodies:
                    R, p = fix[:3, :3] @ R, fix[:3, :3] @ p + fix[:3, 3]
                h.position = tuple(p)
                h.wxyz = tuple(_wxyz(R))


class RcsRobotScene:
    def __init__(self, server, show_gripper: bool = True, show_goal_ghost: bool = True, coupler_offset_m: float = 0.0,
                 coupler_yaw_rad: float = np.pi / 2):
        """coupler_yaw_rad / coupler_offset_m: gripper base relative to the flange (rotation about and offset along
        flange z). The defaults reproduce rcs.GRIPPER_MOUNT_OFFSETS[Robotiq2F85] (+90 deg, flush)."""
        self.coupler_offset_m = coupler_offset_m
        self.coupler_yaw_rad = coupler_yaw_rad
        server.scene.add_grid("/grid", width=2.0, height=2.0)
        self.arm = _Body(server, "/robot", show_gripper, collision_meshes=False)
        # low-poly translucent ghost from the collision meshes (FR3 collision meshes have ~150 vertices per link)
        self.ghost = _Body(server, "/goal_ghost", show_gripper, collision_meshes=True, color=(0.35, 0.75, 0.45),
                           opacity=0.35) if show_goal_ghost else None
        self.ee_frame = server.scene.add_frame("/ee", axes_length=0.08, axes_radius=0.0035)
        self._closure = 0.0
        self._goal_key = "unset"
        if self.ghost is not None:
            self.set_goal(None)

    def _coupler(self) -> np.ndarray:
        c, s = np.cos(self.coupler_yaw_rad), np.sin(self.coupler_yaw_rad)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[2, 3] = self.coupler_offset_m
        return T

    def update(self, q, gripper_closure: float | None = None) -> None:
        """q: 7 measured joint angles [rad]; gripper_closure: 0 = open, 1 = closed (None keeps the last one)."""
        if gripper_closure is not None:
            self._closure = float(gripper_closure)
        self.arm.update(q, self._closure, self._coupler())
        s = self.arm.site
        self.ee_frame.position = tuple(self.arm.d.site_xpos[s])
        self.ee_frame.wxyz = tuple(_wxyz(self.arm.d.site_xmat[s].reshape(3, 3)))

    def set_goal(self, q_goal) -> None:
        """Translucent arm at the goal (None hides it). Called at the refresh rate: only sends changes."""
        if self.ghost is None:
            return
        key = None if q_goal is None else (tuple(np.round(np.asarray(q_goal, dtype=np.float64), 5)),
                                           self.coupler_yaw_rad, self.coupler_offset_m)
        if key == self._goal_key:
            return
        self._goal_key = key
        visible = q_goal is not None
        for h in self.ghost.frames.values():
            h.visible = visible
        if visible:
            self.ghost.update(q_goal, 0.0, self._coupler())
