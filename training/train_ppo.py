"""
training/train_ppo.py
Training loop for the PPO (Proximal Policy Optimization) agent on FamilyLinkEnv.
Run: python training/train_ppo.py

Metrics saved to data/synthetic/ppo_results.json:
  - episodes[]: episode, total_reward, mean_rel_strength, tier2_ratio, engagement
  - losses[]:   combined actor+critic loss per PPO update (one entry per update call)
  - actor_losses[], critic_losses[], entropies[]
  - training_time_s: total wall-clock seconds
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from environment.family_env import FamilyLinkEnv
from agents.ppo_agent       import PPOAgent
from training.config        import ENV_CONFIG
from evaluation.metrics     import MetricsTracker

PPO_CONFIG = {
    "num_episodes":     1000,
    "update_timestep":  400,   # collect this many timesteps before each PPO update
    "lr_actor":         3e-4,
    "lr_critic":        1e-3,
    "gamma":            0.99,
    "K_epochs":         4,
    "eps_clip":         0.2,
    "entropy_coef":     0.01,
}


def train():
    env   = FamilyLinkEnv(ENV_CONFIG)
    # action_dim = feed_pool_size + 2 special actions (Boost, Pass)
    action_dim = ENV_CONFIG["feed_pool_size"] + 2
    state_dim  = ENV_CONFIG["state_dim"]

    agent   = PPOAgent(
        state_dim    = state_dim,
        action_dim   = action_dim,
        lr_actor     = PPO_CONFIG["lr_actor"],
        lr_critic    = PPO_CONFIG["lr_critic"],
        gamma        = PPO_CONFIG["gamma"],
        K_epochs     = PPO_CONFIG["K_epochs"],
        eps_clip     = PPO_CONFIG["eps_clip"],
        entropy_coef = PPO_CONFIG["entropy_coef"],
    )
    tracker = MetricsTracker(label="PPO")

    time_step = 0
    t0        = time.time()

    num_episodes = int(os.environ.get("FAMILYLINK_NUM_EPISODES", PPO_CONFIG["num_episodes"]))
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward       = 0.0
        episode_engagement = 0
        done = False

        while not done:
            action, _ = agent.select_action(state, valid_actions=action_dim)
            next_state, r, done, _, info = env.step(action)

            agent.buffer.rewards.append(r)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            if time_step % PPO_CONFIG["update_timestep"] == 0:
                a_loss, c_loss, ent = agent.update()
                combined = a_loss + c_loss
                tracker.log_loss(combined)
                tracker.log_ppo_sub_losses(a_loss, c_loss, ent)

            state = next_state
            total_reward += r
            event = info.get("event", "")
            if event in ("like", "comment", "share"):
                episode_engagement += 1

        tracker.log_episode(ep, total_reward, env, episode_engagement)
        if (ep + 1) % 100 == 0:
            print(f"[PPO] Ep {ep+1:4d} | Reward: {total_reward:7.2f}")

    training_time = time.time() - t0

    os.makedirs("data/synthetic", exist_ok=True)
    torch.save(agent.actor.state_dict(),  "data/synthetic/ppo_actor.pth")
    torch.save(agent.critic.state_dict(), "data/synthetic/ppo_critic.pth")
    tracker.save("data/synthetic/ppo_results.json", training_time=training_time)
    print(f"PPO training complete in {training_time:.1f}s.")


if __name__ == "__main__":
    train()
