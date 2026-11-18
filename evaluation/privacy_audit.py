"""
evaluation/privacy_audit.py
Audit agent behaviour for privacy violations during evaluation.

Run: python evaluation/privacy_audit.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.family_env import FamilyLinkEnv
from agents.dqn_agent       import DQNAgent
from training.config        import ENV_CONFIG, DQN_CONFIG


def audit(model_path="data/synthetic/dqn_model.pth", num_episodes=50):
    env   = FamilyLinkEnv(ENV_CONFIG)
    agent = DQNAgent(ENV_CONFIG["state_dim"],
                     ENV_CONFIG["feed_pool_size"] + 2,
                     DQN_CONFIG)
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("No saved model found — using random policy.")

    violations = 0
    total_steps = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            state, r, done, _, info = env.step(action)
            total_steps += 1
            if info.get("event") == "privacy_violation":
                violations += 1

    rate = 100 * violations / max(total_steps, 1)
    print(f"\n=== Privacy Audit ===")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Total steps   : {total_steps}")
    print(f"  Violations    : {violations}")
    print(f"  Violation rate: {rate:.2f}%")
    print(f"  Status        : {'✅ PASS' if violations == 0 else '❌ FAIL'}")
    return violations


if __name__ == "__main__":
    audit()
