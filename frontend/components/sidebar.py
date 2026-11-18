"""
frontend/components/sidebar.py
Sidebar panel — user profile, algorithm selector, control buttons, and live stats.
"""

import streamlit as st


def render_sidebar(bridge):
    """
    Render the complete sidebar with:
      - Logo / title
      - Current user profile card
      - RL algorithm selector
      - Control buttons (Reset, Step, Run Episode, Train, New Graph)
      - Live training stats (episode, epsilon, reward, violations)
    """

    # ── Title & branding ──────────────────────────────────────────────
    st.sidebar.html('''
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 2.2rem;">🏠</div>
        <div style="
            font-size: 1.35rem; font-weight: 800;
            background: linear-gradient(135deg, #00e5ff, #b388ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 1px;
        ">FamilyLink RL</div>
        <div style="font-size: 0.7rem; color: #6b7280; letter-spacing: 2px; margin-top: 2px;">
            INTELLIGENT FEED OPTIMIZER
        </div>
    </div>
    ''')

    st.sidebar.markdown("---")

    # ── Current user card ─────────────────────────────────────────────
    user_info = bridge.get_current_user_info()
    if user_info:
        st.sidebar.html(f'''
        <div class="glass-card" style="padding: 1rem;">
            <div style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1.5px;">
                Current User
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #e8eaed; margin-top: 4px;">
                👤 {user_info['name']}
            </div>
            <div style="font-size: 0.82rem; color: #9ca3af; margin-top: 4px;">
                🏡 {user_info['family_name']} Family &nbsp;|&nbsp; 
                {user_info['device']} &nbsp;|&nbsp; 
                🕐 Active at {user_info['active_hour']}:00
            </div>
            <div style="font-size: 0.78rem; color: #6b7280; margin-top: 4px;">
                {user_info['age_group']} &nbsp;•&nbsp;
                Family #{user_info['family_id']} &nbsp;•&nbsp;
                {len(user_info['family_members'])} family members
            </div>
        </div>
        ''')
    else:
        st.sidebar.info("No active user — click **Reset Episode** to start.")

    st.sidebar.markdown("---")

    # ── Algorithm Selector ────────────────────────────────────────────
    st.sidebar.html('<div class="section-header">⚙️ RL Algorithm</div>')
    algo = st.sidebar.selectbox(
        "Select algorithm",
        ["PPO (Actor-Critic)", "DQN", "Q-Learning"],
        index=0 if bridge.active_algo == "PPO (Actor-Critic)" else (1 if bridge.active_algo == "DQN" else 2),
        key="algo_select",
        label_visibility="collapsed",
    )
    bridge.active_algo = algo

    st.sidebar.markdown("---")

    # ── Control Buttons ───────────────────────────────────────────────
    st.sidebar.html('<div class="section-header">🎮 Controls</div>')

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reset Episode", key="btn_reset", use_container_width=True):
            bridge.reset_episode()
            st.rerun()
    with col2:
        if st.button("▶️ Run One Step", key="btn_step", use_container_width=True):
            if bridge.is_done:
                bridge.reset_episode()
            bridge.run_step()
            st.rerun()

    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("⏩ Full Episode", key="btn_episode", use_container_width=True):
            bridge.run_full_episode()
            st.rerun()
    with col4:
        if st.button("🧠 Train ×50", key="btn_train", use_container_width=True):
            with st.spinner("Training 50 episodes..."):
                bridge.train_batch(50)
            st.rerun()

    if st.sidebar.button("🌐 New Social Graph", key="btn_graph", use_container_width=True):
        bridge.regenerate_graph()
        st.rerun()

    st.sidebar.markdown("---")

    # ── Live Stats ────────────────────────────────────────────────────
    st.sidebar.html('<div class="section-header">📊 Live Stats</div>')

    status_color = "#69f0ae" if not bridge.is_done else "#ff5252"
    status_text = "🟢 RUNNING" if not bridge.is_done else "🔴 EPISODE DONE"

    st.sidebar.html(f'''
    <div class="glass-card" style="padding: 0.8rem;">
        <table style="width:100%; border-collapse:collapse; font-size: 0.85rem;">
            <tr><td style="color:#6b7280; padding:3px 0;">Episode</td>
                <td style="text-align:right; color:#e8eaed; font-weight:600;">#{bridge.episode}</td></tr>
            <tr><td style="color:#6b7280; padding:3px 0;">Step</td>
                <td style="text-align:right; color:#e8eaed; font-weight:600;">{bridge.step_count}</td></tr>
            <tr><td style="color:#6b7280; padding:3px 0;">Algorithm</td>
                <td style="text-align:right; color:#00e5ff; font-weight:600;">{bridge.active_algo}</td></tr>
            <tr><td style="color:#6b7280; padding:3px 0;">Epsilon (ε)</td>
                <td style="text-align:right; color:#ffab40; font-weight:600;">{bridge.current_epsilon:.4f}</td></tr>
            <tr><td style="color:#6b7280; padding:3px 0;">Cum. Reward</td>
                <td style="text-align:right; color:#69f0ae; font-weight:600;">{bridge.cumulative_reward:.2f}</td></tr>
            <tr><td style="color:#6b7280; padding:3px 0;">Privacy Violations</td>
                <td style="text-align:right; color:#ff5252; font-weight:600;">{bridge.privacy_violations}</td></tr>
            <tr><td style="color:#6b7280; padding:3px 0;">Status</td>
                <td style="text-align:right; color:{status_color}; font-weight:600;">{status_text}</td></tr>
        </table>
    </div>
    ''')
