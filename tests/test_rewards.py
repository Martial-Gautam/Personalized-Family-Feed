"""tests/test_rewards.py — Reward function edge cases."""
from environment.family_env import FamilyLinkEnv
from training.config        import ENV_CONFIG


def test_privacy_violation_reward_is_negative():
    env = FamilyLinkEnv(ENV_CONFIG)
    penalty = env.cfg["privacy_violation_penalty"]
    assert penalty < 0, "Privacy violation penalty must be negative"


def test_episode_reward_is_float():
    env = FamilyLinkEnv(ENV_CONFIG)
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert isinstance(reward, float)


def test_reward_within_reasonable_range():
    env = FamilyLinkEnv(ENV_CONFIG)
    env.reset()
    for _ in range(10):
        _, r, done, _, _ = env.step(env.action_space.sample())
        assert -10.0 <= r <= 10.0, f"Reward {r} out of expected range"
        if done:
            break
