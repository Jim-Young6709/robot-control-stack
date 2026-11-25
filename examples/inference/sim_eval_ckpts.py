"""Blind sim evaluation of DRP checkpoints through the full deployment stack.

For every checkpoint a `drp` vlagents server (github.com/Jim-Young6709/vlagents, branch drp-agent) is started, and franka.py's sim env (FR3 + Robotiq,
emulated joint impedance controller, waypoint shaping, noise RGB-D) runs one episode per goal for every gain
preset. Prints a table per (checkpoint, gains) and writes inference_logs/sim_eval/<stamp>/summary.json.

    .venv/bin/python examples/inference/sim_eval_ckpts.py                       # all of ~/drp/ckpt_active
    .venv/bin/python examples/inference/sim_eval_ckpts.py --ckpts ~/drp/ckpt/<exp> --presets kp100_t5
"""

import argparse
import dataclasses
import glob
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import franka  # noqa: E402

HOME = Path.home()
POLICY_PYTHON = HOME / "drp/envs/drp_vlagents/bin/python"

PRESETS = {
    # rcs_fr3 defaults (deoxys joint impedance)
    "kp100_t5": dict(
        kp=[100, 100, 100, 100, 75, 150, 50], kd=[20, 20, 20, 20, 7.5, 15, 5], torque_limit=[5.0] * 7
    ),
    # libfranka joint_impedance_control example gains, default torque limit
    "kp600_t5": dict(
        kp=[600, 600, 600, 600, 250, 150, 50], kd=[50, 50, 50, 50, 30, 25, 15], torque_limit=[5.0] * 7
    ),
    # libfranka example gains with a bit more torque headroom
    "kp600_t12": dict(
        kp=[600, 600, 600, 600, 250, 150, 50],
        kd=[50, 50, 50, 50, 30, 25, 15],
        torque_limit=[12.0, 12.0, 12.0, 12.0, 6.0, 6.0, 6.0],
    ),
}

# collision free IK solutions (FR3 + Robotiq TCP pointing down) inside the FR3 datasheet limits
DEFAULT_GOALS = [
    [0.132, 0.2048, 0.3135, -1.8987, -0.0724, 2.0926, 0.4755],  # tcp (0.55, 0.25, 0.25)
    [-0.132, 0.2048, -0.3135, -1.8987, 0.0724, 2.0926, -0.4755],  # tcp (0.55, -0.25, 0.25)
    [0.1263, -0.1062, 0.5563, -1.7406, 0.0562, 1.6502, 0.6756],  # tcp (0.40, 0.35, 0.45)
    [0.0, 0.3828, 0.0, -1.7612, 0.0, 2.1441, 0.0],  # tcp (0.65, 0.00, 0.20)
    [-0.1553, 0.0095, -0.6331, -1.7545, 0.0057, 1.7622, -0.7895],  # tcp (0.40, -0.40, 0.40)
    [0.0, -0.693, 0.0, -2.0068, 0.0, 1.3138, 0.0],  # tcp (0.30, 0.00, 0.55)
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def episode_metrics(d: dict, goal: np.ndarray, cfg: franka.InferenceConfig) -> dict:
    fps, tol = cfg.fps, cfg.goal_tolerance_rad
    q, tgt, pa = d["q"], d["target"], d["policy_action"]
    err = np.abs(q - goal).max(1)
    reached = np.where(err <= tol)[0]
    path = np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
    straight = np.linalg.norm(q[-1] - q[0])
    # smoothness of the measured motion: joint accelerations / jerk from the measured velocities
    ddq = np.diff(d["dq"], axis=0) * fps if len(q) > 2 else np.zeros((1, 7))
    jerk = np.diff(d["dq"], n=2, axis=0) * fps**2 if len(q) > 3 else np.zeros((1, 7))
    margin = np.minimum(q - franka.FR3_Q_MIN, franka.FR3_Q_MAX - q).min()
    # raw policy output, before any shaping: how far it asks to be from the measured joints, how much that
    # request changes from one step to the next, and whether its direction flips
    lead = pa - q
    lead_n = np.abs(lead).max(1)
    dlead = np.abs(np.diff(lead, axis=0)).max(1) if len(q) > 1 else np.zeros(1)
    norms = np.linalg.norm(lead, axis=1)
    valid = (norms[1:] > 1e-3) & (norms[:-1] > 1e-3)
    cos = np.sum(lead[1:] * lead[:-1], axis=1) / np.maximum(norms[1:] * norms[:-1], 1e-12)
    lo, hi = franka.FR3_Q_MIN + cfg.joint_limit_margin, franka.FR3_Q_MAX - cfg.joint_limit_margin
    vt = np.abs(np.diff(tgt, axis=0)) * fps if len(q) > 1 else np.zeros((1, 7))
    return {
        "lead_med": float(np.median(lead_n)),
        "lead_p95": float(np.percentile(lead_n, 95)),
        "lead_max": float(lead_n.max()),
        "dlead_p95": float(np.percentile(dlead, 95)),
        "dlead_max": float(dlead.max()),
        "reversal_frac": float(np.mean(cos[valid] < 0)) if valid.any() else 0.0,
        "jumpcap_frac": float(np.mean(lead_n > cfg.max_waypoint_jump)),
        "limit_frac": float(np.mean(np.any((pa < lo) | (pa > hi), axis=1))),
        "vlim_frac": float(np.mean(np.any(vt >= 0.98 * cfg.max_joint_speed, axis=1))),
        "jerk_p95": float(np.nanpercentile(np.abs(jerk), 95)),
        "reached": bool(len(reached)),
        "t_reach": float((reached[0] + 1) / fps) if len(reached) else None,
        "final_err": float(err[-1]),
        "start_err": float(np.abs(q[0] - goal).max()),
        "path_eff": float(straight / path) if path > 1e-6 else 0.0,
        "max_dq": float(np.nanmax(np.abs(d["dq"]))),
        "ddq_p95": float(np.nanpercentile(np.abs(ddq), 95)),
        "ddq_max": float(np.nanmax(np.abs(ddq))),
        "collision_steps": int(d["collision"].sum()) if "collision" in d else -1,
        "min_margin": float(margin),
        "latency_ms": float(1e3 * d["latency"].mean()),
        "steps": len(q),
    }


def run_preset(
    base: franka.InferenceConfig, port: int, preset: dict, goals, max_steps: int, traj_dir: Path | None = None
) -> list[dict]:
    cfg = dataclasses.replace(base, vlagents_model="drp", vlagents_port=port, goal_joints=goals, **preset)
    env = franka.get_env(cfg, franka.RobotPlatform.SIMULATION, headless=True, realtime=False)
    ctrl = franka.ModelInference(env, cfg, franka.RobotPlatform.SIMULATION, max_steps=max_steps)
    ctrl.max_episodes = len(goals)
    episodes = []

    def keep(tag, reason):
        episodes.append(({k: np.stack(v) for k, v in ctrl.log.rows.items() if v}, reason))

    ctrl.log.save = keep
    ctrl.submit_command("e")
    with env:
        ctrl.loop()
    ctrl.remote_agent.close()
    out = []
    for i, (d, reason) in enumerate(episodes):
        goal = np.asarray(goals[i % len(goals)])
        if traj_dir is not None:
            traj_dir.mkdir(parents=True, exist_ok=True)
            np.savez(traj_dir / f"ep{i:02d}.npz", **d, goal=goal, end=reason)
        out.append(dict(episode_metrics(d, goal, cfg), end=reason))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="*", default=None, help="checkpoint dirs (default ~/drp/ckpt_active/*)")
    parser.add_argument("--presets", nargs="*", default=list(PRESETS), choices=list(PRESETS))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-traj", action="store_true", help="save every episode as npz under <out>/traj/")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    ckpts = args.ckpts or sorted(glob.glob(str(HOME / "drp/ckpt_active/*")))
    base = franka.load_inference_config()
    out_dir = Path(args.out or f"inference_logs/sim_eval/{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ckpt in ckpts:
        name = os.path.basename(os.path.normpath(ckpt))
        port = free_port()
        log = open(out_dir / f"server_{name}.log", "w")
        kwargs = json.dumps({"checkpoint_path": ckpt})
        server = subprocess.Popen(
            [str(POLICY_PYTHON), "-m", "vlagents", "start-server", "drp", "--host", "localhost", "--port", str(port),
             "--kwargs", kwargs],
            stdout=log,
            stderr=log,
        )
        try:
            time.sleep(2.0)
            for preset in args.presets:
                traj_dir = out_dir / "traj" / name / preset if args.save_traj else None
                res = run_preset(base, port, PRESETS[preset], DEFAULT_GOALS, args.max_steps, traj_dir)
                summary.setdefault(name, {})[preset] = res
                n_reach = sum(r["reached"] for r in res)
                t = [r["t_reach"] for r in res if r["reached"]]
                print(
                    f"{name[:60]:60s} {preset:10s} reached {n_reach}/{len(res)} "
                    f"t_reach {np.mean(t) if t else float('nan'):5.1f}s  "
                    f"final_err max {max(r['final_err'] for r in res):.3f}  "
                    f"path_eff {np.mean([r['path_eff'] for r in res]):.2f} | RAW: "
                    f"lead med {np.median([r['lead_med'] for r in res]):.3f} p95 {max(r['lead_p95'] for r in res):.3f} "
                    f"max {max(r['lead_max'] for r in res):.3f}  "
                    f"dlead p95 {max(r['dlead_p95'] for r in res):.3f} max {max(r['dlead_max'] for r in res):.3f}  "
                    f"rev {100 * np.mean([r['reversal_frac'] for r in res]):4.1f}%  "
                    f"cap {100 * max(r['jumpcap_frac'] for r in res):.1f}% lim {100 * max(r['limit_frac'] for r in res):.1f}% "
                    f"vlim {100 * max(r['vlim_frac'] for r in res):.1f}% | EXEC: "
                    f"max_dq {max(r['max_dq'] for r in res):.2f}  "
                    f"ddq95 {np.mean([r['ddq_p95'] for r in res]):4.2f} ddqmax {max(r['ddq_max'] for r in res):5.2f}  "
                    f"jerk95 {np.mean([r['jerk_p95'] for r in res]):5.1f}  "
                    f"coll {sum(max(r['collision_steps'], 0) for r in res):3d}  "
                    f"margin {min(r['min_margin'] for r in res):.2f}  "
                    f"lat {np.mean([r['latency_ms'] for r in res]):.1f}ms",
                    flush=True,
                )
        finally:
            server.terminate()
            server.wait(timeout=30)
            log.close()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
