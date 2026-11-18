"""
training/train_q.py
Training loop for the Tabular Q-Learning baseline agent.
Run: python training/train_q.py

Metrics saved to data/synthetic/q_results.json:
  - episodes[]: episode, total_reward, mean_rel_strength, tier2_ratio, engagement
  - losses[]:   TD-error (|target - Q(s,a)|) per update step
  - training_time_s: total wall-clock seconds
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from environment.family_env import FamilyLinkEnv
from agents.q_agent         import QAgent
from training.config        import ENV_CONFIG, Q_CONFIG
from evaluation.metrics     import MetricsTracker


def train():
    env     = FamilyLinkEnv(ENV_CONFIG)
    agent   = QAgent(ENV_CONFIG["state_dim"],
                     ENV_CONFIG["feed_pool_size"] + 2,
                     Q_CONFIG)
    tracker = MetricsTracker(label="Q-Learning")

    t0 = time.time()

    num_episodes = int(os.environ.get("FAMILYLINK_NUM_EPISODES", Q_CONFIG["num_episodes"]))
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        episode_engagement = 0
        done = False

        while not done:
            action            = agent.select_action(state)
            next_state, r, done, _, info = env.step(action)
            # Compute TD error BEFORE update for logging
            key      = agent._discretise(state)
            next_key = agent._discretise(next_state)
            q        = agent._get_q(key)
            next_q   = agent._get_q(next_key)
            target   = r if done else r + agent.cfg["gamma"] * np.max(next_q)
            td_error = abs(target - q[action])
            tracker.log_loss(td_error)

            agent.update(state, action, r, next_state, done)
            state        = next_state
            total_reward += r
            event = info.get("event", "")
            if event in ("like", "comment", "share"):
                episode_engagement += 1

        tracker.log_episode(ep, total_reward, env, episode_engagement)
        if (ep + 1) % 50 == 0:
            print(f"[Q] Ep {ep+1:4d} | Reward: {total_reward:7.2f} | eps: {agent.epsilon:.3f}")

    training_time = time.time() - t0
    tracker.save("data/synthetic/q_results.json", training_time=training_time)
    print(f"Q-Learning training complete in {training_time:.1f}s.")


if __name__ == "__main__":
    train()
