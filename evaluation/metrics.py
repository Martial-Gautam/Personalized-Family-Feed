"""
evaluation/metrics.py
Tracks and stores per-episode and per-step training metrics for all agents.

Metrics tracked:
  - Cumulative reward per episode
  - Engagement events per episode (likes, comments, shares)
  - Relationship strength snapshot (mean across all pairs)
  - Feed diversity (Tier-2 posts ratio in last candidate feed)
  - Loss per optimizer step (Huber/MSE for DQN, TD-error for Q, combined for PPO)
  - PPO-specific: actor loss, critic loss, entropy (per update)
  - Total training time in seconds
"""
import json
import numpy as np


class MetricsTracker:
    def __init__(self, label="Agent"):
        self.label         = label
        self.episodes      = []
        self.losses        = []          # scalar per optimizer step / PPO update
        self.actor_losses  = []          # PPO only
        self.critic_losses = []          # PPO only
        self.entropies     = []          # PPO only

    # ── Per-step loss ─────────────────────────────────────────────────
    def log_loss(self, loss_value):
        self.losses.append(round(float(loss_value), 6))

    def log_ppo_sub_losses(self, actor_loss, critic_loss, entropy):
        """Log PPO-specific sub-loss components."""
        self.actor_losses.append(round(float(actor_loss),  6))
        self.critic_losses.append(round(float(critic_loss), 6))
        self.entropies.append(round(float(entropy),        6))

    # ── Per-episode metrics ───────────────────────────────────────────
    def log_episode(self, ep_num, total_reward, env, engagement=0):
        # Mean relationship strength across all pairs
        strengths     = list(env.graph.relationship.values())
        mean_strength = float(np.mean(strengths)) if strengths else 0.0

        # Tier-2 ratio in last candidate feed
        posts       = env.candidate_posts
        tier2_ratio = sum(1 for p in posts if p.tier == 2) / max(len(posts), 1)

        self.episodes.append({
            "episode":          ep_num,
            "total_reward":     round(total_reward,    4),
            "mean_rel_strength": round(mean_strength,  4),
            "tier2_ratio":      round(tier2_ratio,     4),
            "engagement":       int(engagement),
        })

    # ── Derived stats ─────────────────────────────────────────────────
    def engagement_improvement(self, baseline_reward):
        """Compute % lift in mean reward over a baseline."""
        if not self.episodes:
            return 0.0
        mean_rew = np.mean([e["total_reward"] for e in self.episodes])
        return round(100 * (mean_rew - baseline_reward) / max(abs(baseline_reward), 1e-6), 2)

    # ── Persistence ───────────────────────────────────────────────────
    def save(self, path, training_time=None):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "label":            self.label,
            "training_time_s":  round(training_time, 2) if training_time is not None else None,
            "episodes":         self.episodes,
            "losses":           self.losses,
            "actor_losses":     self.actor_losses,
            "critic_losses":    self.critic_losses,
            "entropies":        self.entropies,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Metrics saved to {path}")
