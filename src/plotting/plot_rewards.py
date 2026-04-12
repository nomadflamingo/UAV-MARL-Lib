"""Plot consolidated eval/mean_reward for pursuers and evaders separately."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

PURSUER_IDS = {"0", "1"}
EVADER_IDS = {"2", "3"}


def load_eval_data(run_dir):
    """Load evaluations.npz for all agents found in run_dir."""
    agents = {}
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("eval_agent_"):
            npz_path = os.path.join(run_dir, name, "evaluations.npz")
            if os.path.exists(npz_path):
                agent_id = name.replace("eval_agent_", "")
                data = np.load(npz_path)
                agents[agent_id] = {
                    "timesteps": data["timesteps"],
                    "results": data["results"],  # (N, n_eval_episodes)
                }
    return agents


def make_continuous_timesteps(timesteps):
    """Convert per-iteration timesteps (with resets) into a continuous global axis."""
    continuous = np.zeros_like(timesteps, dtype=np.int64)
    offset = 0
    continuous[0] = timesteps[0]
    for i in range(1, len(timesteps)):
        if timesteps[i] <= timesteps[i - 1]:
            offset += timesteps[i - 1]
        continuous[i] = offset + timesteps[i]
    return continuous


def find_iteration_boundaries(timesteps):
    """Return indices where a new FP iteration starts."""
    boundaries = [0]
    for i in range(1, len(timesteps)):
        if timesteps[i] <= timesteps[i - 1]:
            boundaries.append(i)
    return boundaries


def plot_group(ax, agents, agent_ids, group_name, smooth, continuous_ts_ref, boundaries):
    """Plot averaged eval mean reward for a group of agents on a given axis."""
    # Collect per-agent mean rewards, truncate to shortest
    group_means = []
    x = None
    for agent_id, data in agents.items():
        if agent_id not in agent_ids:
            continue
        if x is None:
            x = make_continuous_timesteps(data["timesteps"])
        group_means.append(data["results"].mean(axis=1))

    min_len = min(len(m) for m in group_means)
    group_means = np.stack([m[:min_len] for m in group_means])
    x = x[:min_len]
    avg = group_means.mean(axis=0)
    std = group_means.std(axis=0)

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        avg = np.convolve(avg, kernel, mode="same")
        std = np.convolve(std, kernel, mode="same")

    line, = ax.plot(x, avg, label=group_name, alpha=0.9)
    ax.fill_between(x, avg - std, avg + std, alpha=0.2, color=line.get_color())

    for b in boundaries[1:]:
        ax.axvline(continuous_ts_ref[b], color="gray", linestyle="--", alpha=0.3, linewidth=0.7)


def main():
    parser = argparse.ArgumentParser(description="Plot consolidated eval rewards")
    parser.add_argument(
        "--run_dir",
        default="results/pe/save-pursuit_evasion-04.01.2026_22.29",
        help="Path to a training run directory",
    )
    parser.add_argument("--smooth", type=int, default=5, help="Rolling mean window size (0=off)")
    parser.add_argument("--output", default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    agents = load_eval_data(args.run_dir)
    if not agents:
        print(f"No eval data found in {args.run_dir}")
        return

    first_agent = next(iter(agents.values()))
    raw_ts = first_agent["timesteps"]
    boundaries = find_iteration_boundaries(raw_ts)
    continuous_ts = make_continuous_timesteps(raw_ts)

    run_name = os.path.basename(args.run_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    plot_group(ax1, agents, PURSUER_IDS, "Pursuers", args.smooth, continuous_ts, boundaries)
    ax1.set_ylim(-50, 250)
    ax1.set_ylabel("Eval Mean Reward")
    ax1.set_title(f"Pursuers — {run_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plot_group(ax2, agents, EVADER_IDS, "Evaders", args.smooth, continuous_ts, boundaries)
    ax2.set_ylim(-150, 50)
    ax2.set_xlabel("Total Training Timesteps (across all FP iterations)")
    ax2.set_ylabel("Eval Mean Reward")
    ax2.set_title(f"Evaders — {run_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
