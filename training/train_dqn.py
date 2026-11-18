"""
training/train_dqn.py
Training loop for the Deep Q-Network (DQN) agent.
Run: python training/train_dqn.py

Metrics saved to data/synthetic/dqn_results.json:
  - episodes[]: episode, total_reward, mean_rel_strength, tier2_ratio, engagement
  - losses[]:   loss value per optimizer step
  - training_time_s: total wall-clock seconds
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.family_env import FamilyLinkEnv
from agents.dqn_agent       import DQNAgent
from training.config        import ENV_CONFIG, DQN_CONFIG
from evaluation.metrics     import MetricsTracker


def train():
    env     = FamilyLinkEnv(ENV_CONFIG)
    agent   = DQNAgent(ENV_CONFIG["state_dim"],
                       ENV_CONFIG["feed_pool_size"] + 2,
                       DQN_CONFIG)
    tracker = MetricsTracker(label="DQN")

    t0 = time.time()

    num_episodes = int(os.environ.get("FAMILYLINK_NUM_EPISODES", DQN_CONFIG["num_episodes"]))
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        episode_engagement = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, r, done, _, info = env.step(action)
            agent.store(state, action, r, next_state, done)
            loss  = agent.learn()
            if loss is not None:
                tracker.log_loss(loss)
            state = next_state
            total_reward += r
            event = info.get("event", "")
            if event in ("like", "comment", "share"):
                episode_engagement += 1

        tracker.log_episode(ep, total_reward, env, episode_engagement)
        if (ep + 1) % 100 == 0:
            print(f"[DQN] Ep {ep+1:4d} | Reward: {total_reward:7.2f} | eps: {agent.epsilon:.3f}")

    training_time = time.time() - t0
    agent.save("data/synthetic/dqn_model.pth")
    tracker.save("data/synthetic/dqn_results.json", training_time=training_time)
    print(f"DQN training complete in {training_time:.1f}s.")


if __name__ == "__main__":
    train()
