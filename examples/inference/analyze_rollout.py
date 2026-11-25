"""Summarize and plot trajectory logs written by franka.py (inference_logs/*.npz).

    .venv/bin/python examples/inference/analyze_rollout.py inference_logs/<file>.npz [--plot out.png]
    .venv/bin/python examples/inference/analyze_rollout.py --latest            # newest log
"""

import argparse
import glob
import json
import os

import numpy as np

import rcs

FR3_LIMITS = rcs.ROBOTS[rcs.common.RobotType.FR3].joint_limits
# FR3 datasheet joint velocity limits (rad/s)
FR3_DQ_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])


def summarize(path: str, plot: str | None):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    q, tgt, pa = d["q"], d["target"], d["policy_action"]
    dq = d["dq"] if "dq" in d else np.full_like(q, np.nan)
    fps = meta.get("fps", 30)
    t = np.arange(len(q)) / fps
    np.set_printoptions(precision=3, suppress=True)
    print(f"== {path}  ({len(q)} steps, {len(q) / fps:.1f} s at {fps} Hz)")
    print("meta:", {k: meta[k] for k in ("episode", "model", "camera_source", "goal_joints") if k in meta})
    print("start q            ", q[0])
    print("final q            ", q[-1])
    if meta.get("goal_joints") is not None:
        goal = np.asarray(meta["goal_joints"])
        err = np.abs(q - goal).max(1)
        reach = np.where(err < 0.02)[0]
        print("goal               ", goal)
        print(f"final |q-goal|max   {err[-1]:.4f} rad, reached(<0.02) at {reach[0] / fps:.2f}s" if len(reach)
              else f"final |q-goal|max   {err[-1]:.4f} rad, never within 0.02 rad")
    step = np.abs(np.diff(pa, axis=0))
    print("max |policy Δ|/step", step.max(0), f"(clip {meta.get('max_rel_mov_joints', float('nan')):.4f})")
    print("policy action - q  ", np.abs(pa - q).max(0), "(how far the policy commands ahead of the state)")
    print("max |target-q|     ", np.abs(tgt - q).max(0))
    print("max |dq|           ", np.nanmax(np.abs(dq), 0), "(FR3 limit", FR3_DQ_MAX, ")")
    margin = np.minimum(q - FR3_LIMITS[0], FR3_LIMITS[1] - q).min(0)
    print("min joint-limit margin", margin)
    jerk = np.diff(pa, n=3, axis=0) * fps**3
    print("policy action jerk p95 (rad/s^3)", np.percentile(np.abs(jerk), 95, axis=0))
    if "latency" in d:
        lat = 1e3 * d["latency"].reshape(-1)
        print(f"policy latency ms: mean {lat.mean():.1f}  p95 {np.percentile(lat, 95):.1f}  max {lat.max():.1f}")
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
        for j in range(7):
            ax = axes[j, 0]
            ax.plot(t, q[:, j], label="q (measured)")
            ax.plot(t, tgt[:, j], "--", label="target (after clip)")
            ax.plot(t, pa[:, j], ":", label="policy action")
            if meta.get("goal_joints") is not None:
                ax.axhline(meta["goal_joints"][j], color="k", lw=0.8, alpha=0.5, label="goal")
            ax.set_ylabel(f"j{j + 1} [rad]")
            axes[j, 1].plot(t, dq[:, j])
            axes[j, 1].axhline(FR3_DQ_MAX[j], color="r", lw=0.8)
            axes[j, 1].axhline(-FR3_DQ_MAX[j], color="r", lw=0.8)
            axes[j, 1].set_ylabel(f"dq{j + 1} [rad/s]")
        axes[0, 0].legend(loc="upper right", fontsize=7)
        axes[-1, 0].set_xlabel("t [s]")
        axes[-1, 1].set_xlabel("t [s]")
        fig.suptitle(os.path.basename(path))
        fig.tight_layout()
        fig.savefig(plot, dpi=110)
        print("plot saved to", plot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--plot", default=None, help="png path (only for a single log)")
    parser.add_argument("--log-dir", default="inference_logs")
    args = parser.parse_args()
    paths = args.paths
    if args.latest or not paths:
        paths = sorted(glob.glob(os.path.join(args.log_dir, "*.npz")))[-1:]
    for p in paths:
        summarize(p, args.plot if len(paths) == 1 else None)


if __name__ == "__main__":
    main()
