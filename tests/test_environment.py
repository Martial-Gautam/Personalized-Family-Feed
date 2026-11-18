"""tests/test_environment.py — Environment unit tests."""
import pytest
from environment.family_env import FamilyLinkEnv
from training.config        import ENV_CONFIG


@pytest.fixture
def env():
    return FamilyLinkEnv(ENV_CONFIG)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset()
    assert obs.shape == (ENV_CONFIG["state_dim"],)
    assert all(0.0 <= v <= 1.0 for v in obs)


def test_step_returns_correct_shape(env):
    env.reset()
    obs, reward, done, _, info = env.step(0)
    assert obs.shape == (ENV_CONFIG["state_dim"],)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_episode_terminates(env):
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps <= ENV_CONFIG["max_steps_per_episode"]
