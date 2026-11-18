"""tests/test_access_control.py — Privacy boundary enforcement tests."""
import pytest
from environment.social_graph   import SocialGraph
from environment.access_control import AccessControl
from environment.post           import Post
from training.config            import ENV_CONFIG


@pytest.fixture
def graph():
    return SocialGraph(ENV_CONFIG)


@pytest.fixture
def access(graph):
    return AccessControl(graph)


def test_family_member_can_view_family_only_post(graph, access):
    for fam in graph.families.values():
        if len(fam) >= 2:
            viewer, poster = fam[0], fam[1]
            post = Post(999, poster.user_id, poster.family_id, tier=1, poster_trust=0.7)
            post.visibility = "family-only"
            assert access.can_view(viewer, post)
            break


def test_non_family_cannot_view_family_only_post(graph, access):
    families = list(graph.families.values())
    if len(families) >= 2:
        viewer = families[0][0]
        poster = families[1][0]
        post   = Post(998, poster.user_id, poster.family_id, tier=2, poster_trust=0.3)
        post.visibility = "family-only"
        assert not access.can_view(viewer, post)


def test_no_privacy_violation_in_filtered_feed(graph, access):
    from environment.simulator import InteractionSimulator
    sim = InteractionSimulator(graph, ENV_CONFIG)
    user = graph.sample_user()
    feed = sim.generate_feed(user, access, pool_size=10)
    for post in feed:
        assert not access.audit_action(user, post), \
            f"Privacy violation: user {user.user_id} shown post {post.post_id}"
