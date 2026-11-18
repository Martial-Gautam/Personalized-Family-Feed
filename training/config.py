"""
training/config.py
Centralised hyperparameters and environment configuration.
"""

ENV_CONFIG = {
    # Social graph
    "num_families":           10,
    "family_size_range":      (3, 8),
    "max_trusted_connections": 3,

    # Episode
    "feed_pool_size":          10,
    "max_steps_per_episode":   20,

    # State dimension
    # = user_context(3) + history(10) + relationship_vec(10) + post_encoding(6) + tier_flag(1) + fatigue(1)
    "state_dim":               31,

    # Reward weights
    "w1": 0.35,   # explicit feedback
    "w2": 0.35,   # implicit engagement
    "w3": 0.15,   # fatigue penalty
    "w4": 0.15,   # relationship bonus
    "fatigue_alpha": 0.10,
    "privacy_violation_penalty": -5.0,
}

DQN_CONFIG = {
    "lr":               1e-3,
    "gamma":            0.99,
    "epsilon_start":    1.0,
    "epsilon_end":      0.05,
    "epsilon_decay":    0.995,
    "batch_size":       64,
    "replay_capacity":  10_000,
    "target_sync_freq": 100,    # sync target network every N steps
    "hidden_dims":      [128, 64],
    "num_episodes":     1000,
}

Q_CONFIG = {
    "lr":               0.1,
    "gamma":            0.99,
    "epsilon_start":    1.0,
    "epsilon_end":      0.05,
    "epsilon_decay":    0.995,
    "num_episodes":     500,
    "num_bins":         5,      # bins for state discretisation
}
