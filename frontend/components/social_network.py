"""
frontend/components/social_network.py
Social Network Panel — displays family members, trusted connections,
and relationship strengths with visual progress bars and radar chart.
"""

import streamlit as st
from frontend.charts.analytics import relationship_radar


def _strength_class(value):
    """Return CSS class name based on relationship strength."""
    if value >= 0.65:
        return "high"
    elif value >= 0.35:
        return "medium"
    return "low"


def _strength_label(value):
    """Human-readable strength label."""
    if value >= 0.8:
        return "Very Strong 💚"
    elif value >= 0.65:
        return "Strong 💙"
    elif value >= 0.45:
        return "Moderate 💛"
    elif value >= 0.25:
        return "Weak 🧡"
    return "Very Weak ❤️‍🩹"


def render_social_network(bridge):
    """
    Render the Social Network panel with:
      - Current user identity
      - Family member list with relationship strength bars
      - Trusted connections with badges
      - Relationship radar chart
    """
    user_info = bridge.get_current_user_info()
    if not user_info:
        st.info("Start an episode to see the social network.")
        return

    st.html('<div class="section-header">🌐 Social Network Graph</div>')

    # ── Current User Banner ───────────────────────────────────────────
    st.html(f'''
    <div class="glass-card" style="text-align:center; padding: 1.2rem;">
        <div style="font-size: 2.5rem;">👤</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #e8eaed; margin-top: 0.3rem;">
            {user_info['name']}
        </div>
        <div style="margin-top: 0.3rem;">
            <span class="badge badge-family">🏡 {user_info['family_name']} Family</span>
            &nbsp;
            <span class="badge badge-trusted">👥 {len(user_info['trusted_connections'])} Trusted</span>
        </div>
    </div>
    ''')

    # ── Family Members ────────────────────────────────────────────────
    col_fam, col_trust = st.columns(2)

    with col_fam:
        st.html('<div class="section-header" style="font-size: 1rem;">👨‍👩‍👧‍👦 Family Members</div>')

        for member in user_info["family_members"]:
            s = member["strength"]
            cls = _strength_class(s)
            pct = int(s * 100)
            st.html(f'''
            <div class="post-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #e8eaed; font-weight: 600;">{member['name']}</span>
                        <br>
                        <span style="font-size: 0.75rem; color: #6b7280;">{member['relation']}</span>
                    </div>
                    <div style="text-align: right; min-width: 80px;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: #e8eaed;">{s:.2f}</span>
                        <br>
                        <span style="font-size: 0.7rem; color: #9ca3af;">{_strength_label(s)}</span>
                    </div>
                </div>
                <div class="strength-bar-bg">
                    <div class="strength-bar-fill {cls}" style="width: {pct}%;"></div>
                </div>
            </div>
            ''')

    with col_trust:
        st.html('<div class="section-header" style="font-size: 1rem;">🤝 Trusted Connections</div>')

        if user_info["trusted_connections"]:
            for conn in user_info["trusted_connections"]:
                s = conn["strength"]
                cls = _strength_class(s)
                pct = int(s * 100)
                st.html(f'''
                <div class="post-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #e8eaed; font-weight: 600;">{conn['name']}</span>
                            <br>
                            <span class="badge badge-trusted" style="font-size: 0.65rem;">TIER 2</span>
                        </div>
                        <div style="text-align: right; min-width: 80px;">
                            <span style="font-size: 0.85rem; font-weight: 600; color: #e8eaed;">{s:.2f}</span>
                        </div>
                    </div>
                    <div class="strength-bar-bg">
                        <div class="strength-bar-fill {cls}" style="width: {pct}%;"></div>
                    </div>
                </div>
                ''')
        else:
            st.html('''
            <div style="color: #6b7280; font-size: 0.85rem; padding: 1rem; text-align: center;">
                No trusted connections for this user.
            </div>
            ''')

    # ── Relationship Radar ────────────────────────────────────────────
    st.markdown("---")
    all_connections = user_info["family_members"] + user_info["trusted_connections"]
    if all_connections:
        fig = relationship_radar(all_connections, user_info["name"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
