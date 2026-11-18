"""
frontend/app.py
Main Streamlit application entry point.
Run with: streamlit run frontend/app.py
"""

import sys
import os
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.styles.theme import inject_css
from frontend.simulation.bridge import SimulationBridge
from frontend.components.sidebar import render_sidebar
from frontend.components.social_network import render_social_network
from frontend.components.feed_panel import render_feed_panel, render_interaction_panel
from frontend.components.analytics_panel import render_analytics_panel

# ── Streamlit Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FamilyLink RL | Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject Theme ────────────────────────────────────────────────────────
st.html(inject_css())

# ── Initialise Simulation Bridge in Session State ───────────────────────
if "bridge" not in st.session_state:
    st.session_state.bridge = SimulationBridge()

bridge = st.session_state.bridge

# ── Render Layout ───────────────────────────────────────────────────────
render_sidebar(bridge)

# Main Dashboard Container
st.html('''
<div style="padding: 1rem 0 2rem 0; text-align: center;">
    <h1 style="color: #e8eaed; font-weight: 800; font-size: 2.2rem; letter-spacing: -0.5px;">
        FamilyLink RL Demo Dashboard
    </h1>
    <p style="color: #9ca3af; font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
        Interactive reinforcement learning environment for privacy-aware, family-centric social network recommendation.
    </p>
</div>
''')

# Tabs for Main View
tab1, tab2 = st.tabs(["📱 Simulation View", "📈 RL Analytics"])

with tab1:
    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_social_network(bridge)
    with col_right:
        render_feed_panel(bridge)
        render_interaction_panel(bridge)

with tab2:
    render_analytics_panel(bridge)
