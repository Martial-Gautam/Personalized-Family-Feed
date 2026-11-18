"""
environment/access_control.py
Content visibility enforcement for the two-tier privacy model.

Rules:
  "family-only"     -> same family only
  "trusted-visible" -> same family OR a Tier-1/2 connection
  No public feed exists in this platform.
"""


class AccessControl:
    def __init__(self, social_graph):
        self.graph = social_graph

    def can_view(self, user, post):
        if user.user_id == post.poster_id:
            return True
        if post.visibility == "family-only":
            return user.family_id == self.graph.all_users[post.poster_id].family_id
        if post.visibility == "trusted-visible":
            tier = self.graph.get_tier(user.user_id, post.poster_id)
            return tier in (1, 2)
        return False

    def filter_feed(self, user, posts):
        return [p for p in posts if self.can_view(user, p)]

    def audit_action(self, user, post):
        """Returns True if showing this post to user IS a privacy violation."""
        return not self.can_view(user, post)
