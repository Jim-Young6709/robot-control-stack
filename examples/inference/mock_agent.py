"""Stand-in vlagents policy server to test the RCS <-> vlagents <-> env pipeline without a real model.

The agent "joint_goal_mock" moves the joints toward SingleObs.info["goal_joints"] with a capped joint speed
and returns absolute joint targets (the same contract as the real policy server).
  --mode closed_loop: every target is measured joints + capped step (re-anchored on the measurement each step)
  --mode open_loop:   targets follow an internally integrated reference that starts at the measured joints
  --mode drp_like:    every target is measured joints + a fixed-size step (--drp-step, max-norm) toward the goal,
                      like the DRP checkpoints whose chunk[0] sits ~0.2 rad ahead regardless of the goal distance
On the first step of each episode it logs what it received (camera names, shapes, dtypes, value ranges) so the
observation format can be checked.

    .venv/bin/python examples/inference/mock_agent.py --port 20000 --max-speed 0.3
"""

import argparse
import logging

import numpy as np
import rpyc
from vlagents import register_agent
from vlagents.policies.interface import Act, Agent, Obs
from vlagents.server import AgentService

logger = logging.getLogger("joint_goal_mock")


class JointGoalMock(Agent):
    def __init__(
        self,
        max_speed: float = 0.3,
        fps: float = 30.0,
        chunk: int = 1,
        mode: str = "closed_loop",
        drp_step: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(default_checkpoint_path="", **kwargs)
        self.max_step = max_speed / fps
        self.chunk = chunk
        self.mode = mode
        self.drp_step = drp_step
        self._ref: np.ndarray | None = None

    def act(self, obs: Obs) -> Act:
        super().act(obs)
        robot_name, single_obs = self._require_single_arm(obs)
        q = np.asarray(single_obs.joints, dtype=np.float64)
        goal = single_obs.info.get("goal_joints")
        goal = q if goal is None else np.asarray(goal, dtype=np.float64)
        if single_obs.info.get("step", 0) == 0:
            cams = {
                k: (v.shape, v.dtype.name, float(np.min(v)), float(np.max(v))) for k, v in single_obs.cameras.items()
            }
            logger.info(
                "episode %s start: cameras=%s joints=%s gripper=%s goal=%s",
                single_obs.info.get("episode"),
                cams,
                np.round(q, 3).tolist(),
                single_obs.gripper,
                np.round(goal, 3).tolist(),
            )
        if self.mode == "drp_like":
            d = goal - q
            n = float(np.max(np.abs(d)))
            step = d if n <= self.drp_step else d * (self.drp_step / n)
            return self._chunk_act(robot_name, np.stack([q + step] * self.chunk))
        if self.mode == "open_loop":
            if single_obs.info.get("step", 0) == 0 or self._ref is None:
                self._ref = q.copy()
            cur = self._ref
        else:
            cur = q.copy()
        targets = []
        for _ in range(self.chunk):
            cur = cur + np.clip(goal - cur, -self.max_step, self.max_step)
            targets.append(cur.copy())
        if self.mode == "open_loop":
            self._ref = targets[0]
        return self._chunk_act(robot_name, np.stack(targets))


register_agent("joint_goal_mock", JointGoalMock)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--max-speed", type=float, default=0.3, help="rad/s per joint")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--mode", choices=["closed_loop", "open_loop", "drp_like"], default="closed_loop")
    parser.add_argument("--drp-step", type=float, default=0.2)
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
    agent = JointGoalMock(
        max_speed=args.max_speed, fps=args.fps, chunk=args.chunk, mode=args.mode, drp_step=args.drp_step
    )
    service = AgentService(agent, "joint_goal_mock")
    with service:
        server = rpyc.ThreadedServer(
            service,
            port=args.port,
            hostname=args.host,
            protocol_config={"allow_pickle": True, "allow_public_attrs": True},
        )
        server.start()


if __name__ == "__main__":
    main()
