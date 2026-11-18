"""
frontend/styles/theme.py
Injects custom CSS into Streamlit for dark glassmorphism theme.

The CSS covers:
  - Global background gradient
  - Glassmorphism card panels
  - Sidebar styling
  - Custom progress bars
  - Typography
  - Animations (pulse, glow, fade-in)
  - Badge and tag styles
"""


def inject_css():
    """Return a <style> block string to inject via st.markdown(unsafe_allow_html)."""
    return """
<style>
/* ── Google Font Import ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ──────────────────────────────────────────────────── */
:root {
    --bg-primary: #0a0a1a;
    --bg-secondary: #0f0f23;
    --bg-card: rgba(20, 20, 50, 0.65);
    --bg-card-hover: rgba(30, 30, 70, 0.75);
    --border-glass: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(0, 229, 255, 0.25);
    --text-primary: #e8eaed;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-cyan: #00e5ff;
    --accent-purple: #b388ff;
    --accent-green: #69f0ae;
    --accent-orange: #ffab40;
    --accent-red: #ff5252;
    --accent-pink: #ff80ab;
    --shadow-glass: 0 8px 32px rgba(0, 0, 0, 0.4);
    --font-main: 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ── Global Background ───────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 30%, #0f0f23 70%, #0a0a2e 100%) !important;
    font-family: var(--font-main) !important;
}

/* ── Hide default Streamlit footer and header ────────────────────────── */
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 10, 30, 0.95) 0%, rgba(15, 15, 40, 0.95) 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-secondary) !important;
    font-size: 0.92rem;
}

/* ── Glass Card ──────────────────────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow-glass);
    animation: fadeInUp 0.5s ease-out;
}
.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.08);
}

/* ── Section Headers ─────────────────────────────────────────────────── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Metric Cards ────────────────────────────────────────────────────── */
.metric-card {
    background: rgba(20, 20, 50, 0.5);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-glow);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* ── Strength Bar ────────────────────────────────────────────────────── */
.strength-bar-bg {
    height: 8px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    overflow: hidden;
    margin: 4px 0;
}
.strength-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.strength-bar-fill.high   { background: linear-gradient(90deg, #69f0ae, #00e5ff); }
.strength-bar-fill.medium { background: linear-gradient(90deg, #ffab40, #ff9100); }
.strength-bar-fill.low    { background: linear-gradient(90deg, #ff5252, #d50000); }

/* ── Badges ──────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-family {
    background: rgba(0, 229, 255, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(0, 229, 255, 0.3);
}
.badge-trusted {
    background: rgba(179, 136, 255, 0.15);
    color: var(--accent-purple);
    border: 1px solid rgba(179, 136, 255, 0.3);
}
.badge-blocked {
    background: rgba(255, 82, 82, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(255, 82, 82, 0.3);
}
.badge-pass {
    background: rgba(105, 240, 174, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(105, 240, 174, 0.3);
}

/* ── Post Card ───────────────────────────────────────────────────────── */
.post-card {
    background: rgba(20, 20, 50, 0.4);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: all 0.3s ease;
}
.post-card:hover {
    border-color: rgba(179, 136, 255, 0.3);
}
.post-card.selected {
    border: 2px solid var(--accent-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.15);
}
.post-card.filtered {
    opacity: 0.45;
    border-color: var(--accent-red);
    background: rgba(255, 82, 82, 0.05);
}

/* ── Decision Flow ───────────────────────────────────────────────────── */
.flow-step {
    background: rgba(20, 20, 50, 0.5);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: var(--text-primary);
    animation: fadeInUp 0.3s ease-out;
}
.flow-arrow {
    text-align: center;
    color: var(--accent-cyan);
    font-size: 1.2rem;
    line-height: 1.2;
    opacity: 0.6;
}

/* ── Privacy Shield ──────────────────────────────────────────────────── */
.privacy-shield {
    background: linear-gradient(135deg, rgba(105, 240, 174, 0.08), rgba(0, 229, 255, 0.05));
    border: 1px solid rgba(105, 240, 174, 0.25);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.privacy-shield.fail {
    background: linear-gradient(135deg, rgba(255, 82, 82, 0.08), rgba(255, 128, 171, 0.05));
    border-color: rgba(255, 82, 82, 0.25);
}

/* ── Streamlit Button Overrides ──────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.12), rgba(179, 136, 255, 0.12)) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.3s ease !important;
    font-family: var(--font-main) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.25), rgba(179, 136, 255, 0.25)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.2) !important;
    transform: translateY(-1px);
}

/* ── Streamlit Select Boxes ──────────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(20, 20, 50, 0.6) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Streamlit Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(20, 20, 50, 0.4);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 229, 255, 0.12) !important;
    color: var(--accent-cyan) !important;
}

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(0, 229, 255, 0.2); }
    50%      { box-shadow: 0 0 20px rgba(0, 229, 255, 0.4); }
}
.pulse-glow {
    animation: pulse-glow 2s ease-in-out infinite;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 229, 255, 0.5); }

/* ── Expander styling ────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(20, 20, 50, 0.4) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
</style>
"""
