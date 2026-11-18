import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Actor Network for PPO.
        Outputs a probability distribution over discrete actions using softmax.
        """
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state, valid_actions=None):
        """
        state: Tensor of shape (batch_size, state_dim)
        valid_actions: int or tensor of valid action counts to mask
        """
        logits = self.net(state)
        
        # If we need to restrict to a certain number of candidates
        if valid_actions is not None:
            # Mask invalid actions with a very large negative number
            mask = torch.ones_like(logits, dtype=torch.bool)
            if isinstance(valid_actions, int):
                mask[:, :valid_actions] = False
            else:
                for i, v in enumerate(valid_actions):
                    mask[i, :v] = False
            logits = logits.masked_fill(mask, -1e9)
            
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist
