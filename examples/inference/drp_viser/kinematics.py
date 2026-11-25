"""FR3 kinematics for the panel, built on RCS.

FK: rcs.common.Pin on the RCS FR3 MJCF (the same model and frame the RCS env builds for the robot).
Jacobian: Pinocchio (pin==3.7.0, an rcs-core dependency) on the same MJCF; Pin itself exposes only forward/inverse.
IK: damped least squares with an optional null-space pull toward a posture and an optional position-only mode.
"""

import numpy as np
import pinocchio

import rcs
from rcs import common

FR3 = rcs.ROBOTS[common.RobotType.FR3]
ROBOTIQ_TCP = rcs.GRIPPER_TCP_OFFSETS[common.GripperType("Robotiq2F85")]  # flange -> TCP, +0.1493 m along z
Q_HOME = rcs.HOME_POSITIONS["FR3_DROID"]


def _rotvec(R: np.ndarray) -> np.ndarray:
    return common.Pose(rotation=np.ascontiguousarray(R), translation=np.zeros(3)).rotvec()[3:]


class RcsKinematics:
    def __init__(self, joint_limits: np.ndarray | None = None, q_home: np.ndarray = Q_HOME,
                 tcp_offset: common.Pose = ROBOTIQ_TCP):
        """joint_limits: 2x7 (lower; upper) for clamping IK solutions, e.g. the FR3 datasheet limits minus the
        joint_limit_margin that the waypoint shaper and goal check use. Default: the RCS MJCF ranges."""
        self.limits = np.asarray(FR3.joint_limits if joint_limits is None else joint_limits, dtype=np.float64)
        self.q_home = np.asarray(q_home, dtype=np.float64)
        self.tcp_offset = tcp_offset
        # own instance: Pin keeps an internal pinocchio::Data, so do not share it across threads or with env.robot.get_ik()
        self._pin = common.Pin(FR3.mjcf_model_path, FR3.attachment_site, urdf=False, joint_limits=self.limits)
        self._model = pinocchio.buildModelFromMJCF(FR3.mjcf_model_path)
        site = self._model.getFrameId(FR3.attachment_site)
        parent = self._model.frames[site]
        self._model.addFrame(pinocchio.Frame("tcp", parent.parentJoint, site,
                                             parent.placement * pinocchio.SE3(tcp_offset.pose_matrix()),
                                             pinocchio.FrameType.OP_FRAME))
        self._frame_id = {"flange": site, "tcp": self._model.getFrameId("tcp")}
        self._data = self._model.createData()

    # ------------------------------------------------------------------------------------------- FK / Jacobian
    def pose(self, q, frame: str = "flange") -> np.ndarray:
        """4x4 pose in the robot base frame of the flange (attachment_site) or the TCP (flange * tcp_offset)."""
        offset = self.tcp_offset if frame == "tcp" else common.Pose()
        return self._pin.forward(np.asarray(q, dtype=np.float64).reshape(7), offset).pose_matrix()

    def flange_pose(self, q) -> np.ndarray:
        return self.pose(q, "flange")

    def tcp_pose(self, q) -> np.ndarray:
        return self.pose(q, "tcp")

    def jacobian(self, q, frame: str = "flange") -> np.ndarray:
        """6x7 [v; w] Jacobian expressed in the base frame."""
        return pinocchio.computeFrameJacobian(self._model, self._data, np.asarray(q, dtype=np.float64).reshape(7),
                                              self._frame_id[frame], pinocchio.LOCAL_WORLD_ALIGNED)

    # --------------------------------------------------------------------------------------------- rotations
    @staticmethod
    def wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
        return tuple(common.Pose(rotation=np.ascontiguousarray(R, dtype=np.float64)).rotation_q_wxyz())

    @staticmethod
    def matrix(wxyz) -> np.ndarray:
        w, x, y, z = (float(v) for v in wxyz)
        return common.Pose(quaternion=np.array([x, y, z, w])).rotation_m()  # rcs quaternions are xyzw

    # -------------------------------------------------------------------------------------------------- IK
    def ik(self, target_position, target_rotation=None, q_init=None, q_nullspace=None, iterations: int = 300,
           tolerance: float = 1e-4, damping: float = 0.02, nullspace_gain: float = 0.3, rotation_weight: float = 0.6,
           max_step: float = 0.2, frame: str = "flange") -> tuple[np.ndarray, bool]:
        """Returns (q, ok). target_rotation None solves position only. nullspace_gain is the per-iteration pull of
        the redundant direction toward q_nullspace (default: home), faded out as the task converges."""
        lo, hi = self.limits
        q_null = self.q_home if q_nullspace is None else np.asarray(q_nullspace, dtype=np.float64).reshape(7)
        q = np.clip((q_null if q_init is None else np.asarray(q_init, dtype=np.float64)).reshape(7).copy(), lo, hi)
        p_t = np.asarray(target_position, dtype=np.float64).reshape(3)
        R_t = None if target_rotation is None else np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        w = np.array([1.0, 1.0, 1.0] + [0.0 if R_t is None else rotation_weight] * 3)
        eye6, eye7 = np.eye(6), np.eye(7)
        residual = np.inf
        for _ in range(iterations):
            T = self.pose(q, frame)
            e = np.zeros(6)
            e[:3] = p_t - T[:3, 3]
            if R_t is not None:
                e[3:] = _rotvec(R_t @ T[:3, :3].T)
            e *= w
            residual = float(np.linalg.norm(e))
            if residual < tolerance:
                break
            J = self.jacobian(q, frame) * w[:, None]
            J_pinv = J.T @ np.linalg.solve(J @ J.T + damping**2 * eye6, eye6)
            dq = J_pinv @ e
            if nullspace_gain:
                dq += (eye7 - J_pinv @ J) @ (q_null - q) * (nullspace_gain * min(1.0, residual / 0.02))
            largest = float(np.max(np.abs(dq)))
            if largest > max_step:
                dq *= max_step / largest
            q = np.clip(q + dq, lo, hi)
        return q, bool(residual < tolerance)
