"""
evaluation/plot_results.py
Visualise training metrics: reward curves, relationship strength, tier balance.

Run: python evaluation/plot_results.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def smooth(data, window=20):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot(results_paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for path in results_paths:
        data  = load(path)
        label = data["label"]
        eps   = data["episodes"]
        rewards   = [e["total_reward"]    for e in eps]
        strengths = [e["mean_rel_strength"] for e in eps]
        tier2     = [e["tier2_ratio"]     for e in eps]

        axes[0].plot(smooth(rewards),   label=label)
        axes[1].plot(smooth(strengths), label=label)
        axes[2].plot(smooth(tier2),     label=label)

    axes[0].set_title("Cumulative Reward")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[1].set_title("Mean Relationship Strength")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("R_ij")
    axes[2].set_title("Tier-2 Feed Ratio")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Fraction")

    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("data/synthetic", exist_ok=True)
    plt.savefig("data/synthetic/results.png", dpi=150)
    plt.show()
    print("Plot saved to data/synthetic/results.png")


if __name__ == "__main__":
    paths = [
        "data/synthetic/q_results.json",
        "data/synthetic/dqn_results.json",
    ]
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        plot(existing)
    else:
        print("No results files found. Train first using training/train_q.py or train_dqn.py")
