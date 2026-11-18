"""
evaluation/full_plots.py
===================================================================
Comprehensive training-analysis plots for Q-Learning, DQN, and PPO.

Produces (saved to data/synthetic/plots/):
  Per-model individual plots
  ─────────────────────────
  1.  Cumulative reward vs episodes  (raw + moving average)
  2.  Moving-average reward vs episodes
  3.  Loss vs training steps
  4.  Mean relationship strength vs episode
  5.  Average user engagement vs episode

  Cross-model convergence comparison
  ───────────────────────────────────
  6.  Final reward convergence  (overlay smoothed curves)
  7.  Convergence speed         (episodes to reach 90% of final reward)
  8.  Training stability        (reward std-dev over rolling window)
  9.  Training time             (bar chart)
  10. Final relationship strength (bar chart)
  11. Feed diversity / tier-2 ratio (overlay curves)
  12. Reward variance across episodes (std-dev band overlay)

Run:
  python evaluation/full_plots.py
===================================================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Output directory ──────────────────────────────────────────────
OUT_DIR = os.path.join("data", "synthetic", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────
COLORS = {
    "Q-Learning": "#e07b39",   # warm orange
    "DQN":        "#3a86ff",   # vivid blue
    "PPO":        "#2dc653",   # vivid green
}
ALPHA_RAW = 0.25

# ── Helpers ───────────────────────────────────────────────────────

def smooth(data, window=20):
    """Causal moving average."""
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def smooth_with_x(data, window=20):
    """Return (x_indices, smoothed_values) aligned so x[0] = window//2."""
    data = np.asarray(data, dtype=float)
    s = smooth(data, window)
    offset = len(data) - len(s)
    x = np.arange(offset, len(data))
    return x, s


def rolling_std(data, window=20):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return np.zeros_like(data)
    result = np.zeros(len(data))
    for i in range(window - 1, len(data)):
        result[i] = np.std(data[i - window + 1: i + 1])
    return result[window - 1:]


def convergence_speed(rewards, threshold_frac=0.90, window=20):
    """Return the episode index where rolling mean first exceeds threshold_frac * max."""
    s = smooth(rewards, window)
    target = threshold_frac * np.max(s)
    idxs = np.where(s >= target)[0]
    return int(idxs[0] + (len(rewards) - len(s))) if len(idxs) else len(rewards)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Data loading ──────────────────────────────────────────────────

def load(filename):
    path = os.path.join("data", "synthetic", filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract(data):
    """Pull arrays from a results JSON; works with old and new format."""
    eps         = data.get("episodes", [])
    rewards     = [e["total_reward"]      for e in eps]
    strengths   = [e["mean_rel_strength"] for e in eps]
    tier2       = [e["tier2_ratio"]       for e in eps]
    engagements = [e.get("engagement", 0) for e in eps]
    losses      = data.get("losses", [])
    actor_l     = data.get("actor_losses",  [])
    critic_l    = data.get("critic_losses", [])
    entropies   = data.get("entropies",     [])
    train_time  = data.get("training_time_s", None)
    label       = data.get("label", "?")
    return dict(
        label=label, rewards=rewards, strengths=strengths,
        tier2=tier2, engagements=engagements, losses=losses,
        actor_l=actor_l, critic_l=critic_l, entropies=entropies,
        train_time=train_time,
    )

# ═══════════════════════════════════════════════════════════════════
# PER-MODEL PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_per_model(d):
    label = d["label"]
    color = COLORS.get(label, "#555555")
    rewards     = np.asarray(d["rewards"])
    strengths   = np.asarray(d["strengths"])
    tier2       = np.asarray(d["tier2"])
    engagements = np.asarray(d["engagements"])
    losses      = np.asarray(d["losses"])
    eps_x       = np.arange(len(rewards))

    print(f"\n--- Per-model plots for {label} ---")

    # ── 1+2. Cumulative reward + moving average ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{label} — Reward Curves", fontsize=14, fontweight="bold")

    # Raw + overlay
    axes[0].fill_between(eps_x, 0, rewards, alpha=0.15, color=color)
    axes[0].plot(eps_x, rewards, alpha=ALPHA_RAW, color=color, linewidth=0.8)
    xm, sm = smooth_with_x(rewards, 20)
    axes[0].plot(xm, sm, color=color, linewidth=2, label="MA-20")
    style_ax(axes[0], "Cumulative Reward vs Episodes", "Episode", "Reward")
    axes[0].legend(fontsize=9)

    # Moving average only
    axes[1].plot(xm, sm, color=color, linewidth=2)
    # Std-dev band
    rstd = rolling_std(rewards, 20)
    std_x = np.arange(19, len(rewards))
    axes[1].fill_between(std_x, sm - rstd, sm + rstd,
                          alpha=0.2, color=color, label="±1 σ")
    style_ax(axes[1], "Moving-Average Reward vs Episodes", "Episode", "Smoothed Reward")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    savefig(f"{label.lower().replace('-','_').replace(' ','_')}_reward_curves.png")

    # ── 3. Loss vs training steps ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(losses) > 0:
        step_x = np.arange(len(losses))
        ax.plot(step_x, losses, alpha=ALPHA_RAW, color=color, linewidth=0.6)
        if len(losses) >= 20:
            xl, sl = smooth_with_x(losses, 20)
            ax.plot(xl, sl, color=color, linewidth=2, label="MA-20")
        style_ax(ax, f"{label} — Loss vs Training Steps", "Optimizer Step", "Loss")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No loss data available\n(run training first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=13, color="gray")
        style_ax(ax, f"{label} — Loss vs Training Steps", "Optimizer Step", "Loss")
    plt.tight_layout()
    savefig(f"{label.lower().replace('-','_').replace(' ','_')}_loss.png")

    # ── 4. Mean relationship strength vs episode ──────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps_x, strengths, alpha=ALPHA_RAW, color=color, linewidth=0.8)
    xm2, sm2 = smooth_with_x(strengths, 20)
    ax.plot(xm2, sm2, color=color, linewidth=2, label="MA-20")
    style_ax(ax, f"{label} — Mean Relationship Strength vs Episode",
             "Episode", "Mean Strength [0-1]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(f"{label.lower().replace('-','_').replace(' ','_')}_relationship.png")

    # ── 5. Average user engagement vs episode ────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    if np.any(engagements > 0):
        ax.bar(eps_x, engagements, color=color, alpha=0.35, width=1.0)
        xm3, sm3 = smooth_with_x(engagements, 20)
        ax.plot(xm3, sm3, color=color, linewidth=2, label="MA-20")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5,
                "No engagement data\n(retrain with updated training scripts)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
    style_ax(ax, f"{label} — User Engagement (Likes+Comments+Shares) vs Episode",
             "Episode", "Engagement Events")
    plt.tight_layout()
    savefig(f"{label.lower().replace('-','_').replace(' ','_')}_engagement.png")

    # ── PPO extra: actor/critic loss + entropy ────────────────────
    if label == "PPO" and len(d["actor_l"]) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("PPO — Sub-loss Components", fontsize=13, fontweight="bold")
        for ax, vals, title, ylab in zip(
            axes,
            [d["actor_l"], d["critic_l"], d["entropies"]],
            ["Actor Loss", "Critic Loss", "Policy Entropy"],
            ["Loss", "Loss", "Entropy"],
        ):
            upd_x = np.arange(len(vals))
            ax.plot(upd_x, vals, alpha=0.4, color=color, linewidth=0.8)
            if len(vals) >= 10:
                xu, su = smooth_with_x(vals, min(10, len(vals)))
                ax.plot(xu, su, color=color, linewidth=2)
            style_ax(ax, f"PPO — {title}", "Update Step", ylab)
        plt.tight_layout()
        savefig("ppo_subloss_components.png")


# ═══════════════════════════════════════════════════════════════════
# CONVERGENCE COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════

def plot_comparison(datasets):
    """datasets: list of dicts from extract()"""
    print("\n--- Convergence comparison plots ---")
    labels = [d["label"] for d in datasets]

    # ── 6. Final reward convergence ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for d in datasets:
        c = COLORS.get(d["label"], "#555")
        r = np.asarray(d["rewards"])
        ax.plot(r, alpha=ALPHA_RAW, color=c, linewidth=0.6)
        xm, sm = smooth_with_x(r, 20)
        ax.plot(xm, sm, color=c, linewidth=2.5, label=d["label"])
    style_ax(ax, "Reward Convergence — All Models", "Episode", "Cumulative Reward")
    ax.legend(fontsize=11)
    plt.tight_layout()
    savefig("compare_reward_convergence.png")

    # ── 7. Convergence speed ──────────────────────────────────────
    speeds = [convergence_speed(d["rewards"], 0.90) for d in datasets]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, speeds,
                  color=[COLORS.get(l, "#555") for l in labels],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    style_ax(ax, "Convergence Speed\n(Episodes to reach 90% of peak reward)",
             "Model", "Episodes")
    ax.set_ylim(0, max(speeds) * 1.2)
    plt.tight_layout()
    savefig("compare_convergence_speed.png")

    # ── 8. Reward stability (rolling std-dev) ────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for d in datasets:
        c = COLORS.get(d["label"], "#555")
        r = np.asarray(d["rewards"])
        rstd = rolling_std(r, 20)
        std_x = np.arange(19, len(r))
        ax.plot(std_x, rstd, color=c, linewidth=2, label=d["label"])
    style_ax(ax, "Training Stability — Rolling Reward Std-Dev (window=20)",
             "Episode", "Reward Std-Dev")
    ax.legend(fontsize=11)
    plt.tight_layout()
    savefig("compare_stability.png")

    # ── 9. Training time ─────────────────────────────────────────
    times  = [d.get("train_time") for d in datasets]
    if any(t is not None for t in times):
        safe_times  = [t if t is not None else 0.0 for t in times]
        safe_labels = labels
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(safe_labels, safe_times,
                      color=[COLORS.get(l, "#555") for l in safe_labels],
                      edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, safe_times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
        style_ax(ax, "Training Time Comparison", "Model", "Time (seconds)")
        ax.set_ylim(0, max(safe_times) * 1.2 + 1)
        plt.tight_layout()
        savefig("compare_training_time.png")

    # ── 10. Final relationship strength ───────────────────────────
    final_strengths = [float(np.mean(d["strengths"][-20:])) for d in datasets]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, final_strengths,
                  color=[COLORS.get(l, "#555") for l in labels],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, final_strengths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    style_ax(ax, "Final Mean Relationship Strength (last 20 episodes avg)",
             "Model", "Mean Strength [0-1]")
    ax.set_ylim(0, max(final_strengths) * 1.15)
    plt.tight_layout()
    savefig("compare_relationship_strength.png")

    # ── 11. Feed diversity / tier-2 ratio ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for d in datasets:
        c = COLORS.get(d["label"], "#555")
        t2 = np.asarray(d["tier2"])
        ax.plot(t2, alpha=ALPHA_RAW, color=c, linewidth=0.6)
        xm, sm = smooth_with_x(t2, 20)
        ax.plot(xm, sm, color=c, linewidth=2.5, label=d["label"])
    ax.axhline(y=0.4, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="40% diversity target")
    style_ax(ax, "Feed Diversity — Tier-2 (Trusted) Post Ratio vs Episode",
             "Episode", "Tier-2 Fraction")
    ax.legend(fontsize=11)
    plt.tight_layout()
    savefig("compare_feed_diversity.png")

    # ── 12. Reward variance (std-dev band overlay) ────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for d in datasets:
        c = COLORS.get(d["label"], "#555")
        r = np.asarray(d["rewards"])
        xm, sm = smooth_with_x(r, 20)
        rstd = rolling_std(r, 20)
        std_x = np.arange(19, len(r))
        ax.plot(xm, sm, color=c, linewidth=2.5, label=d["label"])
        ax.fill_between(std_x, sm - rstd, sm + rstd, alpha=0.15, color=c)
    style_ax(ax, "Reward Variance Across Runs — Mean ± Std-Dev (window=20)",
             "Episode", "Reward")
    ax.legend(fontsize=11)
    plt.tight_layout()
    savefig("compare_reward_variance.png")

    # ── Bonus: combined 6-panel summary dashboard ────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("FamilyLink RL — Full Training Dashboard (Q-Learning | DQN | PPO)",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    panel_defs = [
        # (row, col, title, data_key, ylabel, smooth_window)
        (0, 0, "Reward Convergence",          "rewards",     "Reward",         20),
        (0, 1, "Moving-Average Reward",        "rewards",     "Smoothed Reward",20),
        (0, 2, "Reward Variance (±1σ)",        "rewards",     "Reward",         20),
        (1, 0, "Relationship Strength",        "strengths",   "Strength [0-1]", 20),
        (1, 1, "Feed Diversity (Tier-2 Ratio)","tier2",       "Tier-2 Fraction",20),
        (1, 2, "User Engagement",              "engagements", "Engagement",     20),
        (2, 0, "Training Stability (Std-Dev)", "rewards",     "Reward Std-Dev", 20),
    ]

    for row, col, title, key, ylabel, win in panel_defs:
        ax = fig.add_subplot(gs[row, col])
        for d in datasets:
            c = COLORS.get(d["label"], "#555")
            vals = np.asarray(d[key])
            if key == "rewards" and title.startswith("Moving"):
                xm, sm = smooth_with_x(vals, win)
                ax.plot(xm, sm, color=c, linewidth=2, label=d["label"])
            elif key == "rewards" and "Variance" in title:
                xm, sm = smooth_with_x(vals, win)
                rstd  = rolling_std(vals, win)
                std_x = np.arange(win - 1, len(vals))
                ax.plot(xm, sm, color=c, linewidth=2, label=d["label"])
                ax.fill_between(std_x, sm - rstd, sm + rstd, alpha=0.15, color=c)
            elif key == "rewards" and "Stability" in title:
                rstd  = rolling_std(vals, win)
                std_x = np.arange(win - 1, len(vals))
                ax.plot(std_x, rstd, color=c, linewidth=2, label=d["label"])
            else:
                ax.plot(vals, alpha=ALPHA_RAW, color=c, linewidth=0.6)
                xm, sm = smooth_with_x(vals, win)
                ax.plot(xm, sm, color=c, linewidth=2, label=d["label"])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Episode", fontsize=8)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    # Bottom-right 2 cells → bar chart for convergence speed + training time
    ax_speed = fig.add_subplot(gs[2, 1])
    ax_time  = fig.add_subplot(gs[2, 2])

    speeds      = [convergence_speed(d["rewards"], 0.90) for d in datasets]
    bars_s = ax_speed.bar(labels, speeds,
                          color=[COLORS.get(l, "#555") for l in labels],
                          edgecolor="white", width=0.5)
    for bar, val in zip(bars_s, speeds):
        ax_speed.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                      str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_speed.set_title("Convergence Speed (Eps to 90% peak)", fontsize=10, fontweight="bold")
    ax_speed.set_ylabel("Episodes", fontsize=8)
    ax_speed.grid(True, alpha=0.2, linestyle="--")
    ax_speed.spines["top"].set_visible(False)
    ax_speed.spines["right"].set_visible(False)
    ax_speed.tick_params(labelsize=7)

    ttimes = [d.get("train_time") or 0.0 for d in datasets]
    bars_t = ax_time.bar(labels, ttimes,
                         color=[COLORS.get(l, "#555") for l in labels],
                         edgecolor="white", width=0.5)
    for bar, val in zip(bars_t, ttimes):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_time.set_title("Training Time", fontsize=10, fontweight="bold")
    ax_time.set_ylabel("Seconds", fontsize=8)
    ax_time.grid(True, alpha=0.2, linestyle="--")
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.tick_params(labelsize=7)

    # Legend patch
    legend_handles = [Line2D([0], [0], color=COLORS[l], linewidth=3, label=l) for l in labels]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(labels),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01))

    savefig("full_dashboard.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    file_map = {
        "Q-Learning": "q_results.json",
        "DQN":        "dqn_results.json",
        "PPO":        "ppo_results.json",
    }

    datasets = []
    for label, fname in file_map.items():
        raw = load(fname)
        if raw is None:
            print(f"[WARN] {fname} not found — skipping {label}.")
            continue
        d = extract(raw)
        d["label"] = label   # normalise label
        datasets.append(d)

    if not datasets:
        print("No results files found. Run training scripts first:")
        print("  python training/train_q.py")
        print("  python training/train_dqn.py")
        print("  python training/train_ppo.py")
        return

    print(f"\nLoaded results for: {[d['label'] for d in datasets]}")
    print(f"Saving plots to: {OUT_DIR}\n")

    for d in datasets:
        plot_per_model(d)

    if len(datasets) >= 2:
        plot_comparison(datasets)
    else:
        print("\nOnly one model found — skipping comparison plots.")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
