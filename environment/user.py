"""
environment/user.py
User entity with probabilistic behavior model and state encoding.

Tiers:
  Tier 1 — family members     (full trust, high interaction probability)
  Tier 2 — trusted externals  (opt-in, lower interaction probability)
"""
import random
import numpy as np


class User:
    INTERACTION_PROBS = {
        "high":   {"like": 0.35, "comment": 0.20, "share": 0.10, "skip": 0.30, "dislike": 0.05},
        "medium": {"like": 0.25, "comment": 0.10, "share": 0.05, "skip": 0.50, "dislike": 0.10},
        "low":    {"like": 0.10, "comment": 0.05, "share": 0.02, "skip": 0.70, "dislike": 0.13},
    }

    def __init__(self, user_id, family_id):
        self.user_id   = user_id
        self.family_id = family_id
        self.age_group  = random.randint(0, 3)   # 0=child,1=teen,2=adult,3=senior
        self.device     = random.randint(0, 2)   # 0=mobile,1=tablet,2=desktop
        self.active_hour = random.randint(6, 23)
        self.trusted_connections = set()
        self.interaction_history = []            # list of (post_id, event)

    def respond_to(self, post, relationship_strength):
        """Simulate the user's interaction response, based on relationship strength."""
        bucket = "high" if relationship_strength >= 0.65 else \
                 "medium" if relationship_strength >= 0.35 else "low"
        probs  = self.INTERACTION_PROBS[bucket]
        return random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]

    def encode_context(self):
        return np.array([self.age_group / 3.0,
                         self.device    / 2.0,
                         self.active_hour / 23.0], dtype=np.float32)

    def add_to_history(self, post_id, event, maxlen=20):
        self.interaction_history.append((post_id, event))
        if len(self.interaction_history) > maxlen:
            self.interaction_history.pop(0)

    def encode_history(self, maxlen=10):
        event_map = {"like": 1.0, "comment": 0.8, "share": 0.9,
                     "skip": 0.2, "dislike": 0.0}
        recent  = self.interaction_history[-maxlen:]
        encoded = [event_map.get(e, 0.5) for _, e in recent]
        encoded += [0.5] * (maxlen - len(encoded))
        return np.array(encoded, dtype=np.float32)

    def __repr__(self):
        return f"User(id={self.user_id}, family={self.family_id})"
