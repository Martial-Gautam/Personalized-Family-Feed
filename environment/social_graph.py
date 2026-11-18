"""
environment/social_graph.py
Two-tier social graph with dynamic relationship strength.

Tier 1 (Family):  γ = 0.90, init = 0.7, max = 1.0
Tier 2 (Trusted): γ = 0.95, init = 0.3, max = 0.85  (trust ceiling)
"""
import random
import numpy as np
from environment.user import User
from training.config import ENV_CONFIG


class SocialGraph:
    TIER1_INIT   = 0.7
    TIER2_INIT   = 0.3
    TIER2_CAP    = 0.85
    GAMMA_FAM    = 0.90
    GAMMA_TRUST  = 0.95
    INTERACTION_WEIGHTS = {"like": 0.05, "comment": 0.10, "share": 0.15, "dislike": -0.10, "skip": -0.02}

    def __init__(self, config=None):
        self.cfg            = config or ENV_CONFIG
        self.families       = {}    # family_id -> [User]
        self.all_users      = {}    # user_id   -> User
        self.relationship   = {}    # (a,b)     -> float
        self.connection_tier = {}   # (a,b)     -> 1 or 2
        self._build_graph()

    def _key(self, a, b):
        return (min(a, b), max(a, b))

    def _set(self, a, b, value, tier):
        k = self._key(a, b)
        self.relationship[k]    = value
        self.connection_tier[k] = tier

    def _build_graph(self):
        uid = 0
        for fid in range(self.cfg["num_families"]):
            size    = random.randint(*self.cfg["family_size_range"])
            members = [User(uid + i, fid) for i in range(size)]
            for u in members:
                self.all_users[u.user_id] = u
            self.families[fid] = members
            uid += size
            # Tier-1 links
            for i, ua in enumerate(members):
                for ub in members[i+1:]:
                    self._set(ua.user_id, ub.user_id, self.TIER1_INIT, tier=1)

        # Tier-2 links (opt-in cross-family)
        all_ids = list(self.all_users.keys())
        for user in self.all_users.values():
            n = random.randint(0, self.cfg["max_trusted_connections"])
            candidates = [x for x in all_ids
                          if x != user.user_id
                          and self.all_users[x].family_id != user.family_id
                          and self._key(user.user_id, x) not in self.connection_tier]
            for other_id in random.sample(candidates, min(n, len(candidates))):
                self._set(user.user_id, other_id, self.TIER2_INIT, tier=2)
                user.trusted_connections.add(other_id)
                self.all_users[other_id].trusted_connections.add(user.user_id)

    # ── Public API ────────────────────────────────────────────────────
    def get_strength(self, a, b):
        return self.relationship.get(self._key(a, b), 0.0)

    def get_tier(self, a, b):
        return self.connection_tier.get(self._key(a, b), None)

    def are_connected(self, a, b):
        return self._key(a, b) in self.relationship

    def update_relationship(self, a, b, interaction):
        k = self._key(a, b)
        if k not in self.relationship:
            return
        tier  = self.connection_tier[k]
        gamma = self.GAMMA_FAM if tier == 1 else self.GAMMA_TRUST
        delta = self.INTERACTION_WEIGHTS.get(interaction, 0.0)
        new   = gamma * self.relationship[k] + (1 - gamma) * (self.relationship[k] + delta)
        if tier == 2:
            new = min(new, self.TIER2_CAP)
        self.relationship[k] = round(min(new, 1.0), 4)

    def get_connections(self, user_id):
        """Returns [(other_id, tier, strength), ...]"""
        result = []
        for (a, b), strength in self.relationship.items():
            if a == user_id:
                result.append((b, self.connection_tier[(a, b)], strength))
            elif b == user_id:
                result.append((a, self.connection_tier[(a, b)], strength))
        return result

    def get_relationship_vector(self, user_id, max_size=10):
        conns = sorted(self.get_connections(user_id), key=lambda x: -x[2])
        vec   = [c[2] for c in conns[:max_size]]
        vec  += [0.0] * (max_size - len(vec))
        return np.array(vec, dtype=np.float32)

    def sample_user(self):
        return random.choice(list(self.all_users.values()))
