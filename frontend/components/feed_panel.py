"""
frontend/components/feed_panel.py
Feed Recommendation Panel — displays candidate posts with details,
highlights the RL agent's selection, and shows interaction simulation buttons.
"""

import streamlit as st


def render_feed_panel(bridge):
    """
    Render the feed recommendation panel with:
      - Candidate posts with visibility, strength, engagement probability
      - Highlighted agent-selected post
      - Simulated user interaction buttons (Like, Comment, Share, Skip, Dislike)
    """
    st.html('<div class="section-header">📰 Feed Recommendations</div>')

    posts = bridge.get_candidate_posts_info()

    if not posts:
        st.html('''
        <div class="glass-card" style="text-align: center; color: #6b7280;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
            No candidate posts available. Start or reset an episode.
        </div>
        ''')
        return

    # ── Determine which post the agent would select ───────────────────
    last_decision = bridge.decision_log[-1] if bridge.decision_log else None
    selected_idx = last_decision["action"] if last_decision else None

    # ── Content type icons ────────────────────────────────────────────
    type_icons = {
        "photo": "📸", "text": "📝", "event": "🎉", "video": "🎬"
    }

    # ── Render each post ──────────────────────────────────────────────
    for p in posts:
        is_selected = (p["index"] == selected_idx) if selected_idx is not None else False
        card_class = "post-card selected" if is_selected else "post-card"
        icon = type_icons.get(p["content_type"], "📄")

        # Visibility badge
        if p["visibility"] == "family-only":
            vis_badge = '<span class="badge badge-family">🔒 Family Only</span>'
        else:
            vis_badge = '<span class="badge badge-trusted">🌐 Trusted Visible</span>'

        # Tier badge
        tier_badge = '<span class="badge badge-family">T1 Family</span>' if p["tier"] == 1 else \
                     '<span class="badge badge-trusted">T2 Trusted</span>'

        # Selection indicator
        selection_tag = ""
        if is_selected:
            selection_tag = '''
            <div style="
                position: absolute; top: -8px; right: 10px;
                background: linear-gradient(135deg, #00e5ff, #b388ff);
                color: #0a0a1a; font-size: 0.65rem; font-weight: 700;
                padding: 2px 10px; border-radius: 0 0 8px 8px;
                letter-spacing: 1px;
            ">✨ RL SELECTED</div>
            '''

        st.html(f'''
        <div class="{card_class}" style="position: relative;">
            {selection_tag}
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <div style="font-size: 1rem; color: #e8eaed; font-weight: 600;">
                        {icon} {p['description']}
                    </div>
                    <div style="margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap;">
                        {vis_badge} {tier_badge}
                    </div>
                </div>
            </div>
            <div style="
                display: grid; grid-template-columns: repeat(4, 1fr);
                gap: 8px; margin-top: 10px; font-size: 0.78rem;
            ">
                <div class="metric-card" style="padding: 0.5rem;">
                    <div style="color: #00e5ff; font-weight: 700; font-size: 1rem;">
                        {p['relationship_strength']:.2f}
                    </div>
                    <div class="metric-label" style="font-size: 0.65rem;">Rel. Strength</div>
                </div>
                <div class="metric-card" style="padding: 0.5rem;">
                    <div style="color: #69f0ae; font-weight: 700; font-size: 1rem;">
                        {p['engagement_probability']:.0%}
                    </div>
                    <div class="metric-label" style="font-size: 0.65rem;">Engage Prob.</div>
                </div>
                <div class="metric-card" style="padding: 0.5rem;">
                    <div style="color: #ffab40; font-weight: 700; font-size: 1rem;">
                        {p['recency']:.2f}
                    </div>
                    <div class="metric-label" style="font-size: 0.65rem;">Recency</div>
                </div>
                <div class="metric-card" style="padding: 0.5rem;">
                    <div style="color: #b388ff; font-weight: 700; font-size: 1rem;">
                        {p['quality']:.2f}
                    </div>
                    <div class="metric-label" style="font-size: 0.65rem;">Quality</div>
                </div>
            </div>
        </div>
        ''')

    # ── Agent explanation (if a step has been taken) ──────────────────
    if last_decision:
        reward_color = "#69f0ae" if last_decision["reward"] >= 0 else "#ff5252"
        st.html(f'''
        <div class="glass-card" style="border-left: 3px solid #00e5ff;">
            <div class="section-header" style="font-size: 0.95rem;">🤖 Agent Decision Explanation</div>
            <table style="width:100%; font-size: 0.85rem; border-collapse:collapse;">
                <tr><td style="color:#6b7280; padding:4px 0;">Action</td>
                    <td style="color:#e8eaed; font-weight:600;">{last_decision['action_label']}</td></tr>
                <tr><td style="color:#6b7280; padding:4px 0;">Description</td>
                    <td style="color:#e8eaed;">{last_decision['post_description']}</td></tr>
                <tr><td style="color:#6b7280; padding:4px 0;">User Response</td>
                    <td style="color:#e8eaed; font-weight:600;">{last_decision['event'].upper()}</td></tr>
                <tr><td style="color:#6b7280; padding:4px 0;">Reward</td>
                    <td style="color:{reward_color}; font-weight:700; font-size: 1.05rem;">
                        {'+' if last_decision['reward'] >= 0 else ''}{last_decision['reward']:.4f}
                    </td></tr>
                <tr><td style="color:#6b7280; padding:4px 0;">Epsilon (ε)</td>
                    <td style="color:#ffab40;">{last_decision['epsilon']:.4f}</td></tr>
            </table>
        </div>
        ''')
        
        # If PPO was used, show Actor Probs and Critic Value
        if last_decision.get("action_probs") is not None:
            probs = last_decision["action_probs"]
            critic_val = last_decision["critic_value"]
            import math
            entropy = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)
            
            st.html(f'''
            <div class="glass-card" style="border-left: 3px solid #b388ff; margin-top: 10px;">
                <div class="section-header" style="font-size: 0.95rem;">🧠 PPO Actor-Critic Outputs</div>
                <div style="font-size: 0.85rem; color: #e8eaed; margin-bottom: 8px;">
                    <b>Critic Value (V(s)):</b> {critic_val:.4f} &nbsp;|&nbsp; <b>Policy Entropy:</b> {entropy:.4f}
                </div>
                <div style="font-size: 0.75rem; color: #9ca3af; margin-bottom: 4px;">Actor Action Probabilities:</div>
            ''')
            
            for i, p in enumerate(probs):
                if p > 0.01: # only show non-trivial probs
                    width = max(int(p * 100), 1)
                    color = "#b388ff" if i == last_decision['action'] else "#00e5ff"
                    label = f"Post #{i}" if i < len(posts) else "Pass/Boost"
                    st.html(f'''
                    <div style="display: flex; align-items: center; margin-bottom: 4px;">
                        <div style="width: 80px; font-size: 0.75rem; color: #e8eaed;">{label}</div>
                        <div style="flex: 1; background: #1a1a2e; border-radius: 4px; height: 8px; margin: 0 10px;">
                            <div style="width: {width}%; background: {color}; height: 100%; border-radius: 4px;"></div>
                        </div>
                        <div style="width: 40px; font-size: 0.75rem; color: #e8eaed; text-align: right;">{p:.1%}</div>
                    </div>
                    ''')
            st.html('</div>')


def render_interaction_panel(bridge):
    """
    Render user interaction simulation buttons:
    Like, Comment, Share, Skip, Dislike
    Each triggers a step with the corresponding simulated feedback.
    """
    st.html('<div class="section-header">👆 Simulate User Interaction</div>')

    st.html('''
    <div style="font-size: 0.82rem; color: #9ca3af; margin-bottom: 0.8rem;">
        Click <b>▶️ Run One Step</b> in the sidebar to simulate an interaction automatically.
        Alternatively, click any button below to <b>override</b> the simulator and force a specific reaction!
    </div>
    ''')
    cols = st.columns(5)
    actions = [
        ("👍 Like", "like", "#69f0ae"),
        ("💬 Comment", "comment", "#00e5ff"),
        ("🔗 Share", "share", "#b388ff"),
        ("⏭️ Skip", "skip", "#ffab40"),
        ("👎 Dislike", "dislike", "#ff5252"),
    ]
    for col, (label, action, color) in zip(cols, actions):
        with col:
            if st.button(label, key=f"interact_{action}", use_container_width=True):
                # Run a step overriding the environment's simulation
                if bridge.is_done:
                    bridge.reset_episode()
                bridge.run_step(force_event=action)
                st.rerun()

    # Show step rewards bar for current episode
    if bridge.step_rewards:
        from frontend.charts.analytics import step_reward_bar
        fig = step_reward_bar(bridge.step_rewards)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
