import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        """
        Critic Network for PPO.
        Outputs a scalar value V(s) estimating the expected return from the state.
        """
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        """
        state: Tensor of shape (batch_size, state_dim)
        Returns: Tensor of shape (batch_size, 1)
        """
        return self.net(state)
