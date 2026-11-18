"""
environment/simulator.py
Synthetic data generator and feed builder for FamilyLink RL.

Responsibilities:
  - generate_feed(): build the candidate post pool for a user
  - simulate_interaction(): probabilistically choose an interaction event
  - implicit_score(): estimate implicit engagement signal
  - encode_state(): build the full state vector
"""
import random
import numpy as np
from environment.post import Post
from training.config import ENV_CONFIG


class InteractionSimulator:
    _post_counter = 0

    def __init__(self, social_graph, config=None):
        self.graph = social_graph
        self.cfg   = config or ENV_CONFIG

    def _next_post_id(self):
        InteractionSimulator._post_counter += 1
        return InteractionSimulator._post_counter

    def generate_feed(self, user, access_control, pool_size=10):
        """
        Build a candidate post pool for `user`.
        Sourced from Tier-1 (family) and Tier-2 (trusted) connections.
        All posts pass through access_control.filter_feed() before returning.
        """
        raw_posts = []
        connections = self.graph.get_connections(user.user_id)

        # Guarantee at least 60% family posts (Tier-1)
        tier1 = [(oid, tier, strength) for oid, tier, strength in connections if tier == 1]
        tier2 = [(oid, tier, strength) for oid, tier, strength in connections if tier == 2]

        for oid, tier, strength in (tier1 + tier2):
            post = Post(
                post_id         = self._next_post_id(),
                poster_id       = oid,
                poster_family_id= self.graph.all_users[oid].family_id,
                tier            = tier,
                poster_trust    = strength
            )
            raw_posts.append(post)

        allowed = access_control.filter_feed(user, raw_posts)
        return allowed[:pool_size] if len(allowed) >= pool_size else allowed

    def simulate_interaction(self, user, post):
        """Simulate user response to a post given their relationship strength."""
        strength = self.graph.get_strength(user.user_id, post.poster_id)
        event    = user.respond_to(post, strength)
        user.add_to_history(post.post_id, event)
        return event

    def implicit_score(self, user, post):
        """
        Estimate implicit engagement (dwell time proxy).
        Higher relationship strength and better post quality → higher score.
        """
        strength = self.graph.get_strength(user.user_id, post.poster_id)
        score    = 0.5 * strength + 0.3 * post.base_quality + 0.2 * post.recency
        # Add a small random noise
        score   += random.gauss(0, 0.05)
        return float(np.clip(score, 0.0, 1.0))

    def encode_state(self, user, candidate_posts, consecutive_skips):
        """
        Build state vector:
          user_context(3) + history(10) + relationship_vec(10) +
          top_post_encoding(6) + fatigue(1) + tier_balance(1) = 31 dims
        """
        u_ctx   = user.encode_context()                          # (3,)
        history = user.encode_history(maxlen=10)                 # (10,)
        rel_vec = self.graph.get_relationship_vector(user.user_id, max_size=10)  # (10,)

        # Encode the top candidate post (or zeros if no posts)
        if candidate_posts:
            top_post = max(candidate_posts, key=lambda p: p.recency * p.poster_trust)
            post_enc = top_post.encode()                         # (6,)
        else:
            post_enc = np.zeros(6, dtype=np.float32)

        fatigue       = np.array([min(consecutive_skips / 10.0, 1.0)], dtype=np.float32)  # (1,)
        tier2_ratio   = np.array(
            [sum(1 for p in candidate_posts if p.tier == 2) / max(len(candidate_posts), 1)],
            dtype=np.float32
        )  # (1,)

        state = np.concatenate([u_ctx, history, rel_vec, post_enc, fatigue, tier2_ratio])
        return state.astype(np.float32)
