"""
agents/networks.py
PyTorch neural network definitions for DQN.

PolicyNetwork  — online Q-network (trained every step)
TargetNetwork  — periodically synced copy (stable Q targets)
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully-connected Q-network: state -> Q-values for each action."""

    def __init__(self, state_dim, action_dim, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim   = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_networks(state_dim, action_dim, hidden_dims=(128, 64)):
    """Convenience function: returns (policy_net, target_net)."""
    policy = QNetwork(state_dim, action_dim, hidden_dims)
    target = QNetwork(state_dim, action_dim, hidden_dims)
    target.load_state_dict(policy.state_dict())
    target.eval()
    return policy, target
