"""
environment/family_env.py
Core Gym-compatible RL environment for FamilyLink RL.

Observation: state vector (31-dim) built by InteractionSimulator.encode_state()
Action:      index into candidate post pool + 2 special actions (Boost, Pass)
Reward:      explicit + implicit - fatigue - privacy_violation
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment.social_graph  import SocialGraph
from environment.access_control import AccessControl
from environment.simulator     import InteractionSimulator
from training.config           import ENV_CONFIG


class FamilyLinkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        self.cfg    = config or ENV_CONFIG
        self.graph  = SocialGraph(self.cfg)
        self.access = AccessControl(self.graph)
        self.sim    = InteractionSimulator(self.graph, self.cfg)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg["state_dim"],), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.cfg["feed_pool_size"] + 2)

        self.current_user      = None
        self.candidate_posts   = []
        self.step_count        = 0
        self.consecutive_skips = 0
        self.episode_reward    = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_user      = self.graph.sample_user()
        self.candidate_posts   = self.sim.generate_feed(
            self.current_user, self.access, self.cfg["feed_pool_size"]
        )
        self.step_count        = 0
        self.consecutive_skips = 0
        self.episode_reward    = 0.0
        return self._obs(), {}

    def step(self, action, override_event=None):
        reward, info = self._execute(action, override_event)
        self._update_rel(action, info)
        self.episode_reward += reward
        self.step_count     += 1
        done = self.step_count >= self.cfg["max_steps_per_episode"]
        return self._obs(), reward, done, False, info

    def _obs(self):
        return self.sim.encode_state(
            self.current_user, self.candidate_posts, self.consecutive_skips
        )

    def _execute(self, action, override_event=None):
        n = len(self.candidate_posts)

        if action < n:
            post = self.candidate_posts[action]
            # Privacy guard
            if self.access.audit_action(self.current_user, post):
                return self.cfg["privacy_violation_penalty"], {"event": "privacy_violation"}
            if override_event:
                event = override_event
            else:
                event  = self.sim.simulate_interaction(self.current_user, post)
            reward = self._reward(event, post)
            self.consecutive_skips = 0 if event != "skip" else self.consecutive_skips + 1
            return reward, {"event": event, "post_id": post.post_id}

        elif action == n:       # Boost
            return 0.05, {"event": "boost"}
        else:                   # Pass
            self.consecutive_skips += 1
            return -0.05 * self.consecutive_skips, {"event": "pass"}

    def _reward(self, event, post):
        c = self.cfg
        explicit  = {"like": 1.0, "comment": 2.0, "share": 3.0,
                     "dislike": -1.0, "skip": 0.0}.get(event, 0.0)
        implicit  = self.sim.implicit_score(self.current_user, post)
        fatigue   = c["fatigue_alpha"] * (self.consecutive_skips ** 2)
        tier_disc = 1.0 if post.tier == 1 else 0.6 * post.poster_trust
        return round(float(
            (c["w1"] * explicit + c["w2"] * implicit - c["w3"] * fatigue) * tier_disc
        ), 4)

    def _update_rel(self, action, info):
        if info.get("event") in ("like", "comment", "share", "dislike", "skip") and action < len(self.candidate_posts):
            post = self.candidate_posts[action]
            self.graph.update_relationship(
                self.current_user.user_id, post.poster_id, info["event"]
            )

    def render(self):
        print(f"[Step {self.step_count}] User={self.current_user.user_id} "
              f"Reward={self.episode_reward:.2f} Skips={self.consecutive_skips}")
