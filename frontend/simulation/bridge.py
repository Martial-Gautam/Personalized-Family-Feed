"""
frontend/simulation/bridge.py
Bridge layer connecting the Streamlit UI to the existing RL backend.

This module wraps FamilyLinkEnv, DQNAgent, and QAgent so the dashboard
can run live simulation steps, load pre-trained weights, and query
environment state without modifying any backend code.
"""

import sys
import os
import random
import json
import numpy as np

# ── Ensure the project root is on sys.path so backend imports work ────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.family_env   import FamilyLinkEnv
from environment.social_graph import SocialGraph
from environment.post         import Post
from agents.dqn_agent         import DQNAgent
from agents.q_agent           import QAgent
from agents.ppo_agent         import PPOAgent
from training.config          import ENV_CONFIG, DQN_CONFIG, Q_CONFIG

# ── Realistic post description templates ──────────────────────────────────
POST_TEMPLATES = {
    "photo": [
        "🖼️ {name} shared family dinner photos",
        "📸 {name} uploaded vacation pictures",
        "🎂 {name} posted birthday celebration album",
        "🌅 {name} shared a sunset from the terrace",
        "👨‍👩‍👧‍👦 {name} posted a family group photo",
    ],
    "text": [
        "📝 {name} shared a heartfelt message",
        "💬 {name} posted a family update",
        "📖 {name} shared a recipe for dal makhani",
        "✍️ {name} wrote about a childhood memory",
        "📢 {name} announced a family gathering",
    ],
    "event": [
        "🎉 {name} created an event: Family Reunion",
        "🗓️ {name} scheduled a video call — Sunday 6 PM",
        "🎊 {name} is organising a Diwali celebration",
        "🏏 {name} created event: Cricket Match at Park",
        "🎈 {name} planned a surprise birthday party",
    ],
    "video": [
        "🎬 {name} uploaded a cricket highlights reel",
        "📹 {name} shared a cooking tutorial",
        "🎥 {name} posted a festival celebration video",
        "🎞️ {name} shared a travel vlog",
        "📺 {name} uploaded a funny family moments clip",
    ],
}

# ── Synthetic first names for display ─────────────────────────────────────
FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Meera", "Kiran", "Ananya", "Vikram", "Sneha",
    "Arjun", "Deepa", "Ravi", "Kavya", "Suresh", "Pooja", "Amit", "Neha",
    "Sanjay", "Ishita", "Rahul", "Divya", "Arun", "Lakshmi", "Nikhil", "Swati",
    "Manoj", "Ritu", "Varun", "Anjali", "Harsh", "Tanya", "Kunal", "Pallavi",
    "Sachin", "Nisha", "Gaurav", "Shruti", "Rajesh", "Aditi", "Vivek", "Sonali",
]

FAMILY_NAMES = ["Sharma", "Patel", "Gupta", "Iyer", "Khan",
                "Reddy", "Singh", "Verma", "Das", "Joshi"]

RELATION_LABELS = ["Parent", "Sibling", "Spouse", "Child", "Grandparent",
                   "Cousin", "Uncle/Aunt", "Nephew/Niece"]


class SimulationBridge:
    """
    High-level API used by the Streamlit app to drive the RL simulation.
    Wraps environment creation, agent instantiation, and step execution.
    """

    def __init__(self):
        """Initialise environment and both agents (DQN + Q-Learning)."""
        self.env = FamilyLinkEnv(ENV_CONFIG)
        self.action_dim = ENV_CONFIG["feed_pool_size"] + 2

        # Agents
        self.dqn_agent = DQNAgent(
            ENV_CONFIG["state_dim"], self.action_dim, DQN_CONFIG
        )
        self.q_agent = QAgent(
            ENV_CONFIG["state_dim"], self.action_dim, Q_CONFIG
        )
        self.ppo_agent = PPOAgent(
            ENV_CONFIG["state_dim"], action_dim=12
        )

        # Active algorithm selector
        self.active_algo = "DQN"

        # Session state
        self.state = None
        self.episode = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.privacy_violations = 0
        self.is_done = True

        # History tracking for charts
        self.reward_history = []         # per-episode rewards
        self.step_rewards = []           # per-step rewards inside current episode
        self.relationship_history = []   # mean rel strength per episode
        self.epsilon_history = []        # epsilon per episode
        self.tier2_history = []          # tier-2 ratio per episode
        self.decision_log = []           # step-by-step decision trace
        self.privacy_log = []            # privacy audit events

        # User name cache for display
        self._name_cache = {}
        self._assign_names()

        # Try loading pre-trained model
        model_path = os.path.join(PROJECT_ROOT, "data", "synthetic", "dqn_model.pth")
        if os.path.exists(model_path):
            try:
                self.dqn_agent.load(model_path)
                self.dqn_agent.epsilon = DQN_CONFIG["epsilon_end"]  # use greedy for demo
            except Exception:
                pass  # model architecture may differ; ignore gracefully

    # ── Name helpers ──────────────────────────────────────────────────
    def _assign_names(self):
        """Give every synthetic user a human-readable name."""
        name_pool = list(FIRST_NAMES)
        random.shuffle(name_pool)
        idx = 0
        for fid, members in self.env.graph.families.items():
            surname = FAMILY_NAMES[fid % len(FAMILY_NAMES)]
            for u in members:
                first = name_pool[idx % len(name_pool)]
                self._name_cache[u.user_id] = f"{first} {surname}"
                idx += 1

    def get_user_name(self, uid):
        return self._name_cache.get(uid, f"User-{uid}")

    def get_family_surname(self, fid):
        return FAMILY_NAMES[fid % len(FAMILY_NAMES)]

    # ── Active agent shortcut ────────────────────────────────────────
    @property
    def agent(self):
        if self.active_algo == "PPO (Actor-Critic)":
            return self.ppo_agent
        elif self.active_algo == "DQN":
            return self.dqn_agent
        else:
            return self.q_agent

    @property
    def current_epsilon(self):
        if self.active_algo == "PPO (Actor-Critic)":
            return 0.0 # PPO uses stochastic policy, not epsilon-greedy
        return self.agent.epsilon

    # ── Episode lifecycle ────────────────────────────────────────────
    def reset_episode(self):
        """Start a new episode (resets env, picks random user)."""
        self.state, _ = self.env.reset()
        self.is_done = False
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.step_rewards = []
        self.decision_log = []
        self.episode += 1

    def regenerate_graph(self):
        """Create a completely new social graph and reassign names."""
        self.env = FamilyLinkEnv(ENV_CONFIG)
        self._name_cache.clear()
        self._assign_names()
        self.reset_episode()

    # ── Single step ──────────────────────────────────────────────────
    def run_step(self, force_event=None):
        """
        Execute one RL step and return a rich info dict for the UI.
        Returns None if the episode is already done.
        """
        if self.is_done or self.state is None:
            return None

        # Select action via the active agent
        action_probs = None
        critic_value = None
        
        if self.active_algo == "PPO (Actor-Critic)":
            # self.state is already a flat 31-dim state vector from FamilyLinkEnv.
            # Ensure it is a proper 1-D numpy array before passing to the agent.
            global_state = np.asarray(self.state, dtype=np.float32).flatten()
            action, action_probs = self.agent.select_action(global_state, valid_actions=self.action_dim)
            critic_value = self.agent.buffer.values[-1].item() if self.agent.buffer.values else None
        else:
            action = self.agent.select_action(self.state)

        # Snapshot pre-step relationship strengths
        pre_strengths = dict(self.env.graph.relationship)

        # Step the environment
        next_state, reward, done, truncated, info = self.env.step(action, force_event)

        # Track privacy violations
        is_violation = info.get("event") == "privacy_violation"
        if is_violation:
            self.privacy_violations += 1
            self.privacy_log.append({
                "episode": self.episode,
                "step": self.step_count,
                "action": action,
                "detail": "Agent tried to recommend a post that violates privacy rules.",
            })

        # Compute relationship deltas
        post_strengths = dict(self.env.graph.relationship)
        rel_changes = {}
        for k, v in post_strengths.items():
            if pre_strengths.get(k, v) != v:
                rel_changes[k] = round(v - pre_strengths[k], 4)

        # Build decision trace entry
        selected_post = None
        post_description = "N/A"
        n = len(self.env.candidate_posts)

        if action < n:
            selected_post = self.env.candidate_posts[action]
            poster_name = self.get_user_name(selected_post.poster_id)
            templates = POST_TEMPLATES.get(selected_post.content_type, POST_TEMPLATES["text"])
            post_description = random.choice(templates).format(name=poster_name)
        elif action == n:
            post_description = "⬆️ Boost — increase priority of top post"
        else:
            post_description = "⏭️ Pass — skip current feed"

        decision_entry = {
            "step": self.step_count,
            "state_summary": self._summarise_state(self.state),
            "action": action,
            "action_label": self._action_label(action, n),
            "post_description": post_description,
            "event": info.get("event", "unknown"),
            "reward": round(reward, 4),
            "rel_changes": rel_changes,
            "is_privacy_violation": is_violation,
            "epsilon": round(self.current_epsilon, 4),
            "action_probs": action_probs.tolist() if action_probs is not None else None,
            "critic_value": round(critic_value, 4) if critic_value is not None else None,
        }
        self.decision_log.append(decision_entry)

        # Store transition for learning (DQN only in live mode)
        if self.active_algo == "DQN":
            self.dqn_agent.store(self.state, action, reward, next_state, done)
            self.dqn_agent.learn()
        elif self.active_algo == "Q-Learning":
            self.q_agent.update(self.state, action, reward, next_state, done)
        # For PPO, we are just using the buffer during steps but live updating is complex
        # We can optionally call self.ppo_agent.update() at the end of the episode if wanted.

        # Update session accumulators
        self.state = next_state
        self.step_count += 1
        self.cumulative_reward += reward
        self.step_rewards.append(reward)
        self.is_done = done

        # If episode is done, record to history
        if done:
            self.reward_history.append(round(self.cumulative_reward, 4))
            strengths = list(self.env.graph.relationship.values())
            mean_str = round(float(np.mean(strengths)), 4) if strengths else 0.0
            self.relationship_history.append(mean_str)
            self.epsilon_history.append(round(self.current_epsilon, 4))
            posts = self.env.candidate_posts
            t2 = sum(1 for p in posts if p.tier == 2) / max(len(posts), 1)
            self.tier2_history.append(round(t2, 4))

        return decision_entry

    # ── Run full episode ─────────────────────────────────────────────
    def run_full_episode(self):
        """Run an entire episode automatically, returning all decision entries."""
        self.reset_episode()
        entries = []
        while not self.is_done:
            entry = self.run_step()
            if entry:
                entries.append(entry)
        return entries

    # ── Batch training ───────────────────────────────────────────────
    def train_batch(self, num_episodes=50):
        """
        Run N episodes silently and record aggregate stats.
        Used by the 'Start Training' button for quick batch runs.
        """
        for _ in range(num_episodes):
            self.run_full_episode()

    # ── Environment queries ──────────────────────────────────────────
    def get_current_user_info(self):
        """Return a dict of current user details for the sidebar/profile panel."""
        u = self.env.current_user
        if u is None:
            return None
        family_members = self.env.graph.families.get(u.family_id, [])
        connections = self.env.graph.get_connections(u.user_id)

        family_conns = [(cid, tier, s) for cid, tier, s in connections if tier == 1]
        trusted_conns = [(cid, tier, s) for cid, tier, s in connections if tier == 2]

        age_labels = ["Child", "Teen", "Adult", "Senior"]
        device_labels = ["📱 Mobile", "📟 Tablet", "🖥️ Desktop"]

        return {
            "user_id": u.user_id,
            "name": self.get_user_name(u.user_id),
            "family_id": u.family_id,
            "family_name": self.get_family_surname(u.family_id),
            "age_group": age_labels[u.age_group],
            "device": device_labels[u.device],
            "active_hour": u.active_hour,
            "family_members": [
                {
                    "user_id": m.user_id,
                    "name": self.get_user_name(m.user_id),
                    "relation": RELATION_LABELS[m.user_id % len(RELATION_LABELS)],
                    "strength": round(self.env.graph.get_strength(u.user_id, m.user_id), 4),
                }
                for m in family_members if m.user_id != u.user_id
            ],
            "trusted_connections": [
                {
                    "user_id": cid,
                    "name": self.get_user_name(cid),
                    "strength": round(s, 4),
                }
                for cid, tier, s in trusted_conns
            ],
            "family_connections": [
                {
                    "user_id": cid,
                    "name": self.get_user_name(cid),
                    "strength": round(s, 4),
                }
                for cid, tier, s in family_conns
            ],
        }

    def get_candidate_posts_info(self):
        """Return rich info about the current candidate posts for feed display."""
        posts = self.env.candidate_posts
        user = self.env.current_user
        if not posts or user is None:
            return []

        result = []
        for i, p in enumerate(posts):
            poster_name = self.get_user_name(p.poster_id)
            templates = POST_TEMPLATES.get(p.content_type, POST_TEMPLATES["text"])
            description = random.choice(templates).format(name=poster_name)
            strength = self.env.graph.get_strength(user.user_id, p.poster_id)

            # Engagement probability estimate (from user model)
            bucket = "high" if strength >= 0.65 else "medium" if strength >= 0.35 else "low"
            from environment.user import User as UserClass
            probs = UserClass.INTERACTION_PROBS[bucket]
            engage_prob = 1.0 - probs.get("skip", 0.5) - probs.get("dislike", 0.1)

            result.append({
                "index": i,
                "post_id": p.post_id,
                "poster_id": p.poster_id,
                "poster_name": poster_name,
                "description": description,
                "content_type": p.content_type,
                "visibility": p.visibility,
                "tier": p.tier,
                "recency": round(p.recency, 3),
                "quality": round(p.base_quality, 3),
                "relationship_strength": round(strength, 4),
                "engagement_probability": round(engage_prob, 3),
                "poster_trust": round(p.poster_trust, 4),
            })
        return result

    def get_privacy_audit_summary(self):
        """Return a summary dict for the privacy panel."""
        total_steps = max(sum(len(e) for e in [self.decision_log]), 1)
        return {
            "total_violations": self.privacy_violations,
            "total_steps_all_time": self.step_count,
            "violation_rate": round(
                100 * self.privacy_violations / max(self.episode * 20, 1), 2
            ),
            "status": "✅ PASS" if self.privacy_violations == 0 else "⚠️ VIOLATIONS DETECTED",
            "recent_violations": self.privacy_log[-10:],
        }

    # ── Load saved training results for analytics overlay ────────────
    def load_saved_results(self):
        """Load DQN, Q-Learning, and PPO results from data/synthetic/ if available."""
        results = {}
        files = [
            ("dqn_results.json", "DQN"),
            ("q_results.json",   "Q-Learning"),
            ("ppo_results.json", "PPO"),
        ]
        for fname, label in files:
            path = os.path.join(PROJECT_ROOT, "data", "synthetic", fname)
            if os.path.exists(path):
                with open(path) as f:
                    results[label] = json.load(f)
        return results

    # ── Internal helpers ─────────────────────────────────────────────
    def _summarise_state(self, state):
        """Create a human-readable summary of the state vector."""
        if state is None or len(state) < 31:
            return {}
        return {
            "user_context": [round(float(x), 3) for x in state[:3]],
            "history_encoding": [round(float(x), 2) for x in state[3:13]],
            "relationship_vec": [round(float(x), 3) for x in state[13:23]],
            "top_post_enc": [round(float(x), 2) for x in state[23:29]],
            "fatigue": round(float(state[29]), 3),
            "tier2_ratio": round(float(state[30]), 3),
        }

    def _action_label(self, action, n_posts):
        """Human-readable action description."""
        if action < n_posts:
            return f"Recommend Post #{action}"
        elif action == n_posts:
            return "Boost"
        else:
            return "Pass / Skip"
