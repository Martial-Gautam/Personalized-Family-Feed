import matplotlib.pyplot as plt
import json
import os

def load_metrics(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def moving_avg(x, N=20):
    if len(x) < N: return x
    import numpy as np
    return np.convolve(x, np.ones(N)/N, mode='valid')

def plot_comparison():
    # Load metrics from training runs
    ppo_metrics = load_metrics('ppo_metrics.json')
    dqn_metrics = load_metrics('dqn_metrics.json')
    
    if not ppo_metrics:
        print("PPO metrics not found. Run train_ppo.py first.")
        return
        
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN vs PPO Comparison', fontsize=16)
    
    # 1. Rewards Convergence
    axs[0, 0].set_title('Cumulative Rewards')
    axs[0, 0].plot(moving_avg(ppo_metrics['rewards']), label='PPO (Actor-Critic)', color='blue')
    if dqn_metrics:
        axs[0, 0].plot(moving_avg(dqn_metrics['rewards']), label='DQN', color='orange')
    axs[0, 0].set_xlabel('Episodes')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].legend()
    
    # 2. Engagements
    axs[0, 1].set_title('High Engagement Interactions')
    axs[0, 1].plot(moving_avg(ppo_metrics['engagements']), label='PPO', color='green')
    if dqn_metrics:
        axs[0, 1].plot(moving_avg(dqn_metrics['engagements']), label='DQN', color='red')
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Engagements')
    axs[0, 1].legend()
    
    # 3. Relationship Stability
    axs[1, 0].set_title('Network Relationship Strength')
    axs[1, 0].plot(ppo_metrics['relationships'], label='PPO', color='purple')
    if dqn_metrics:
        axs[1, 0].plot(dqn_metrics['relationships'], label='DQN', color='brown')
    axs[1, 0].set_xlabel('Episodes')
    axs[1, 0].set_ylabel('Strength [0-1]')
    axs[1, 0].legend()
    
    # 4. PPO Specific (Entropy/Exploration)
    axs[1, 1].set_title('PPO Policy Entropy vs Time')
    axs[1, 1].plot(ppo_metrics['entropies'], color='black', label='PPO Entropy')
    axs[1, 1].set_xlabel('Updates')
    axs[1, 1].set_ylabel('Entropy')
    axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dqn_vs_ppo_comparison.png')
    print("Comparison plot saved to dqn_vs_ppo_comparison.png")

if __name__ == "__main__":
    plot_comparison()
