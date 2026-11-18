"""
agents/dqn_agent.py
Deep Q-Network (DQN) agent for FamilyLink RL.

Features:
  - ε-greedy exploration with exponential decay
  - Experience replay buffer
  - Target network with periodic hard sync
  - Huber loss for stable training
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.networks    import build_networks
from agents.replay_buffer import ReplayBuffer
from training.config    import DQN_CONFIG


class DQNAgent:
    def __init__(self, state_dim, action_dim, config=None):
        self.cfg        = config or DQN_CONFIG
        self.action_dim = action_dim
        self.epsilon    = self.cfg["epsilon_start"]
        self.steps      = 0

        self.policy_net, self.target_net = build_networks(
            state_dim, action_dim, self.cfg["hidden_dims"]
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg["lr"])
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss
        self.buffer    = ReplayBuffer(self.cfg["replay_capacity"])

    # ── Action selection ──────────────────────────────────────────────
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    # ── Store transition ──────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ── Learn from batch ──────────────────────────────────────────────
    def learn(self):
        if len(self.buffer) < self.cfg["batch_size"]:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.cfg["batch_size"]
        )
        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions)
            target_q     = rewards + self.cfg["gamma"] * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self._decay_epsilon()
        self._maybe_sync_target()

        return loss.item()

    def _decay_epsilon(self):
        self.epsilon = max(
            self.cfg["epsilon_end"],
            self.epsilon * self.cfg["epsilon_decay"]
        )

    def _maybe_sync_target(self):
        if self.steps % self.cfg["target_sync_freq"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="dqn_model.pth"):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
