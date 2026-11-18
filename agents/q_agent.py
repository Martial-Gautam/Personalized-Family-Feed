"""
agents/q_agent.py
Tabular Q-Learning agent (baseline) for FamilyLink RL.
State space is discretised into bins for table lookup.
"""
import numpy as np
import random
from training.config import Q_CONFIG


class QAgent:
    def __init__(self, state_dim, action_dim, config=None):
        self.cfg        = config or Q_CONFIG
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.epsilon    = self.cfg["epsilon_start"]
        self.bins       = self.cfg["num_bins"]

        # Q-table: discretised state -> action_dim values
        # For simplicity, flatten state into a hash key
        self.q_table = {}

    def _discretise(self, state):
        """Bin each state dimension and convert to a hashable key."""
        return tuple(int(np.clip(s, 0.0, 0.9999) * self.bins) for s in state)

    def _get_q(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
        return self.q_table[key]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        key = self._discretise(state)
        return int(np.argmax(self._get_q(key)))

    def update(self, state, action, reward, next_state, done):
        key      = self._discretise(state)
        next_key = self._discretise(next_state)
        q        = self._get_q(key)
        next_q   = self._get_q(next_key)

        target = reward if done else reward + self.cfg["gamma"] * np.max(next_q)
        q[action] += self.cfg["lr"] * (target - q[action])

        self.epsilon = max(
            self.cfg["epsilon_end"],
            self.epsilon * self.cfg["epsilon_decay"]
        )
