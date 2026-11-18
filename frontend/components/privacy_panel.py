"""
frontend/components/privacy_panel.py
Privacy Visualization Panel — shows privacy filtering, audit results,
blocked post examples, and enforcement layers.
"""

import streamlit as st


def render_privacy_panel(bridge):
    """
    Render the Privacy Visualization panel with:
      - Privacy audit status (pass/fail)
      - Violation statistics
      - Blocked post examples with reasons
      - Privacy enforcement layer visualization
      - Audit logs
    """
    st.html('<div class="section-header">🛡️ Privacy Enforcement</div>')

    audit = bridge.get_privacy_audit_summary()
    posts = bridge.get_candidate_posts_info()

    # ── Privacy Status Shield ─────────────────────────────────────────
    shield_class = "privacy-shield" if audit["total_violations"] == 0 else "privacy-shield fail"
    shield_icon = "🛡️" if audit["total_violations"] == 0 else "⚠️"
    status_color = "#69f0ae" if audit["total_violations"] == 0 else "#ff5252"

    st.html(f'''
    <div class="{shield_class}">
        <div style="font-size: 2.5rem;">{shield_icon}</div>
        <div style="
            font-size: 1.3rem; font-weight: 700;
            color: {status_color}; margin-top: 0.3rem;
        ">{audit['status']}</div>
        <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.3rem;">
            Privacy Audit — {bridge.episode} episodes evaluated
        </div>
    </div>
    ''')

    # ── Metrics Row ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.html(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #ff5252;">{audit['total_violations']}</div>
            <div class="metric-label">Total Violations</div>
        </div>
        ''')
    with col2:
        st.html(f'''
        <div class="metric-card">
            <div class="metric-value">{audit['violation_rate']}%</div>
            <div class="metric-label">Violation Rate</div>
        </div>
        ''')
    with col3:
        total_blocked = sum(1 for p in posts if p["visibility"] == "family-only" and p["tier"] == 2)
        st.html(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #69f0ae;">{len(posts) - total_blocked}</div>
            <div class="metric-label">Posts Allowed</div>
        </div>
        ''')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Privacy Enforcement Layers ────────────────────────────────────
    st.html('''
    <div class="glass-card">
        <div class="section-header" style="font-size: 0.95rem;">🔐 Privacy Enforcement Layers</div>
        <div style="font-size: 0.85rem; color: #e8eaed;">
    ''')

    layers = [
        ("Layer 1: Visibility Filter", "family-only", 
         "Posts marked 'family-only' are restricted to same-family members only.",
         "#00e5ff"),
        ("Layer 2: Tier-Based Access", "trusted-visible",
         "Tier-2 posts are visible to trusted connections (opt-in cross-family).",
         "#b388ff"),
        ("Layer 3: Access Control Audit", "audit",
         "Every post recommendation is validated against AccessControl.can_view() before display.",
         "#69f0ae"),
        ("Layer 4: Violation Penalty", "penalty",
         "Privacy violations incur a −5.0 reward penalty, teaching the RL agent to respect privacy.",
         "#ffab40"),
    ]

    for title, tag, description, color in layers:
        st.html(f'''
        <div style="
            display: flex; align-items: center; gap: 12px;
            padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
        ">
            <div style="
                width: 8px; height: 8px; border-radius: 50%;
                background: {color}; box-shadow: 0 0 8px {color};
                flex-shrink: 0;
            "></div>
            <div>
                <div style="font-weight: 600; color: {color}; font-size: 0.88rem;">{title}</div>
                <div style="font-size: 0.78rem; color: #9ca3af;">{description}</div>
            </div>
        </div>
        ''')

    st.html("</div></div>")

    # ── Post Visibility Breakdown ─────────────────────────────────────
    if posts:
        st.html('''
        <div class="section-header" style="font-size: 0.95rem; margin-top: 1rem;">
            📋 Current Feed — Visibility Breakdown
        </div>
        ''')

        for p in posts:
            if p["visibility"] == "family-only":
                icon = "🔒"
                badge = '<span class="badge badge-family">Family Only</span>'
                status_text = "Visible only to family members"
            else:
                icon = "🌐"
                badge = '<span class="badge badge-trusted">Trusted Visible</span>'
                status_text = "Visible to family + trusted connections"

            st.html(f'''
            <div class="post-card" style="padding: 0.7rem 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #e8eaed; font-size: 0.85rem;">
                            {icon} Post #{p['post_id']} — by {p['poster_name']}
                        </span>
                        <br>
                        <span style="font-size: 0.72rem; color: #6b7280;">{status_text}</span>
                    </div>
                    <div>{badge}</div>
                </div>
            </div>
            ''')

    # ── Recent Violation Logs ─────────────────────────────────────────
    if audit["recent_violations"]:
        st.html('''
        <div class="section-header" style="font-size: 0.95rem; margin-top: 1rem;">
            📜 Recent Violation Logs
        </div>
        ''')
        for v in audit["recent_violations"]:
            st.html(f'''
            <div class="post-card filtered" style="padding: 0.7rem 1rem;">
                <div style="color: #ff5252; font-weight: 600; font-size: 0.85rem;">
                    ❌ Episode {v['episode']}, Step {v['step']}
                </div>
                <div style="color: #9ca3af; font-size: 0.78rem; margin-top: 2px;">
                    {v['detail']}
                </div>
            </div>
            ''')
