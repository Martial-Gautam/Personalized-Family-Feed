import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from .rollout_buffer import RolloutBuffer

class PPOAgent:
    def __init__(self, state_dim, action_dim=12, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        self.buffer = RolloutBuffer()
        
        # Initialize Actor and Critic Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Old networks for ratio calculation in clipped objective
        self.actor_old = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old = CriticNetwork(state_dim).to(self.device)
        self.critic_old.load_state_dict(self.critic.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state, valid_actions=None):
        with torch.no_grad():
            state_tensor = torch.tensor(
                np.asarray(state, dtype=np.float32)
            ).unsqueeze(0).to(self.device)
            # Forward pass through old actor
            dist = self.actor_old(state_tensor, valid_actions)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            # Forward pass through old critic
            state_val = self.critic_old(state_tensor)
            
        # Store for PPO update
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(state_val)
        
        return action.item(), dist.probs.squeeze(0).cpu().numpy()
        
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards for stability
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert buffer lists to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_values = torch.squeeze(torch.stack(self.buffer.values, dim=0)).detach().to(self.device)
        
        # Calculate Advantages (Returns - State Values)
        advantages = rewards.detach() - old_values.detach()
        
        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            dist = self.actor(old_states)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(old_states)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values.squeeze(), rewards)
            entropy_loss = -self.entropy_coef * dist_entropy.mean() # Entropy bonus for exploration
            
            loss = actor_loss + critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            entropy_list.append(dist_entropy.mean().item())
            
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        
        # Clear buffer after update
        self.buffer.clear()
        
        return np.mean(actor_loss_list), np.mean(critic_loss_list), np.mean(entropy_list)
