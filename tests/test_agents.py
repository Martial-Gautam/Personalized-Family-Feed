"""tests/test_agents.py — Agent sanity checks."""
import numpy as np
from agents.dqn_agent import DQNAgent
from agents.q_agent   import QAgent
from training.config  import ENV_CONFIG, DQN_CONFIG, Q_CONFIG

STATE_DIM  = ENV_CONFIG["state_dim"]
ACTION_DIM = ENV_CONFIG["feed_pool_size"] + 2


def dummy_transition():
    s  = np.random.rand(STATE_DIM).astype(np.float32)
    ns = np.random.rand(STATE_DIM).astype(np.float32)
    return s, ns


def test_dqn_action_in_range():
    agent = DQNAgent(STATE_DIM, ACTION_DIM, DQN_CONFIG)
    s, _  = dummy_transition()
    a     = agent.select_action(s)
    assert 0 <= a < ACTION_DIM


def test_dqn_learn_runs():
    agent = DQNAgent(STATE_DIM, ACTION_DIM, DQN_CONFIG)
    for _ in range(DQN_CONFIG["batch_size"] + 1):
        s, ns = dummy_transition()
        agent.store(s, 0, 1.0, ns, False)
    loss = agent.learn()
    assert loss is not None and loss >= 0.0


def test_q_agent_update_runs():
    agent = QAgent(STATE_DIM, ACTION_DIM, Q_CONFIG)
    s, ns = dummy_transition()
    a     = agent.select_action(s)
    agent.update(s, a, 1.0, ns, False)
