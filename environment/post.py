"""
environment/post.py
Post entity with content type, privacy visibility tag, and state encoding.

visibility options:
  "family-only"     — only same-family members can see it
  "trusted-visible" — same-family + opted-in trusted connections can see it
"""
import random
import numpy as np


class Post:
    CONTENT_TYPES       = ["photo", "text", "event", "video"]
    TRUSTED_VISIBLE_PROB = 0.30   # chance a Tier-1 post is also trusted-visible

    def __init__(self, post_id, poster_id, poster_family_id, tier, poster_trust=1.0):
        self.post_id          = post_id
        self.poster_id        = poster_id
        self.poster_family_id = poster_family_id
        self.tier             = tier          # 1=family post, 2=trusted-connection post
        self.poster_trust     = poster_trust  # R_ij at feed-generation time

        self.content_type = random.choice(self.CONTENT_TYPES)
        self.visibility   = (
            "trusted-visible" if tier == 2 or random.random() < self.TRUSTED_VISIBLE_PROB
            else "family-only"
        )
        self.recency      = random.uniform(0.3, 1.0)
        self.base_quality = random.uniform(0.4, 1.0)

    def encode(self):
        """Returns np.ndarray shape (6,): [content_onehot x4, recency, tier_flag]"""
        onehot = [0.0] * 4
        onehot[self.CONTENT_TYPES.index(self.content_type)] = 1.0
        return np.array(onehot + [self.recency, float(self.tier - 1)], dtype=np.float32)

    def decay_recency(self, rate=0.05):
        self.recency = max(0.0, self.recency - rate)

    def __repr__(self):
        return f"Post(id={self.post_id}, type={self.content_type}, vis={self.visibility}, tier={self.tier})"
