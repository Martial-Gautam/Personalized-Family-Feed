"""
frontend/charts/analytics.py
Plotly-based chart builders for the RL Analytics panel.

All charts use a consistent dark theme with glassmorphism-inspired styling.
Each function returns a plotly.graph_objects.Figure.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared dark-theme layout defaults ────────────────────────────────────
DARK_BG       = "rgba(15, 15, 35, 0.0)"
GRID_COLOR    = "rgba(255, 255, 255, 0.06)"
FONT_COLOR    = "#c8cad0"
ACCENT_CYAN   = "#00e5ff"
ACCENT_PURPLE = "#b388ff"
ACCENT_GREEN  = "#69f0ae"
ACCENT_ORANGE = "#ffab40"
ACCENT_RED    = "#ff5252"
ACCENT_PINK   = "#ff80ab"
ACCENT_LIME   = "#ccff90"

MODEL_COLORS = {
    "DQN":       ACCENT_CYAN,
    "Q-Learning": ACCENT_PINK,
    "PPO":       ACCENT_GREEN,
}

def _base_layout(**overrides):
    layout = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        margin=dict(l=50, r=30, t=45, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.08)",
            font=dict(size=11)
        ),
    )
    layout.update(overrides)
    return layout


def _smooth(data, window=20):
    """Causal moving average."""
    data = list(data)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid").tolist()


def _rolling_std(data, window=20):
    data = np.asarray(data, dtype=float)
    result = []
    for i in range(window - 1, len(data)):
        result.append(float(np.std(data[i - window + 1: i + 1])))
    return result


# ═══════════════════════════════════════════════════════════════════
# LIVE SESSION CHARTS
# ═══════════════════════════════════════════════════════════════════

def reward_chart(rewards, title="Cumulative Reward per Episode"):
    fig = go.Figure()
    episodes = list(range(len(rewards)))
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, mode="lines", name="Raw Reward",
        line=dict(color=ACCENT_CYAN, width=1), opacity=0.3,
    ))
    smoothed = _smooth(rewards)
    fig.add_trace(go.Scatter(
        x=list(range(len(smoothed))), y=smoothed, mode="lines",
        name="Moving Avg (20)", line=dict(color=ACCENT_CYAN, width=2.5),
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode", yaxis_title="Reward"))
    return fig


def relationship_chart(strengths, title="Mean Relationship Strength"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(strengths))), y=strengths, mode="lines",
        name="Mean R_ij", line=dict(color=ACCENT_PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(179,136,255,0.12)",
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode", yaxis_title="Strength"))
    return fig


def epsilon_chart(epsilons, title="Exploration Rate Decay"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(epsilons))), y=epsilons, mode="lines",
        name="Epsilon", line=dict(color=ACCENT_ORANGE, width=2.5),
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode", yaxis_title="Epsilon"))
    return fig


def tier2_chart(ratios, title="Tier-2 (Trusted) Post Ratio"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(ratios))), y=ratios, mode="lines",
        name="Tier-2 Ratio", line=dict(color=ACCENT_GREEN, width=1), opacity=0.3,
    ))
    smoothed = _smooth(ratios, 10)
    fig.add_trace(go.Scatter(
        x=list(range(len(smoothed))), y=smoothed, mode="lines",
        name="Smoothed", line=dict(color=ACCENT_GREEN, width=2.5),
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode", yaxis_title="Fraction"))
    return fig


def step_reward_bar(step_rewards, title="Step Rewards (Current Episode)"):
    colors = [ACCENT_GREEN if r >= 0 else ACCENT_RED for r in step_rewards]
    fig = go.Figure(go.Bar(
        x=list(range(len(step_rewards))), y=step_rewards,
        marker_color=colors, name="Reward",
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Step", yaxis_title="Reward"))
    return fig


def relationship_radar(connections, user_name="User"):
    if not connections:
        return go.Figure()
    names = [c["name"].split()[0] for c in connections[:8]]
    values = [c["strength"] for c in connections[:8]]
    names.append(names[0])
    values.append(values[0])
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=names, fill="toself",
        fillcolor="rgba(0,229,255,0.15)", line=dict(color=ACCENT_CYAN, width=2),
        name=user_name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR,
                            tickfont=dict(size=10, color=FONT_COLOR)),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11, color=FONT_COLOR)),
        ),
        **_base_layout(title=f"Relationship Radar — {user_name}"),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# OFFLINE COMPARISON CHARTS  (all 3 models)
# ═══════════════════════════════════════════════════════════════════

def _extract(data):
    """Pull arrays from a results JSON (both old and new format)."""
    eps         = data.get("episodes", [])
    rewards     = [e.get("total_reward", 0)       for e in eps]
    strengths   = [e.get("mean_rel_strength", 0)  for e in eps]
    tier2       = [e.get("tier2_ratio", 0)         for e in eps]
    engagements = [e.get("engagement", 0)          for e in eps]
    losses      = data.get("losses", [])
    return dict(rewards=rewards, strengths=strengths, tier2=tier2,
                engagements=engagements, losses=losses,
                train_time=data.get("training_time_s"),
                label=data.get("label", "?"))


# ── 1. Generic metric comparison (reward / relationship / tier2) ──
def comparison_chart(saved_results, metric="total_reward",
                     title="All Models — Comparison"):
    fig = go.Figure()
    for label, data in saved_results.items():
        color = MODEL_COLORS.get(label, ACCENT_CYAN)
        d = _extract(data)
        key_map = {"total_reward": "rewards", "mean_rel_strength": "strengths",
                   "tier2_ratio": "tier2", "engagement": "engagements"}
        values = d.get(key_map.get(metric, "rewards"), [])
        smoothed = _smooth(values)
        fig.add_trace(go.Scatter(
            x=list(range(len(values))), y=values, mode="lines",
            name=f"{label} (raw)", line=dict(color=color, width=1), opacity=0.2,
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(smoothed))), y=smoothed, mode="lines",
            name=f"{label}", line=dict(color=color, width=2.5),
        ))
    y_label_map = {"total_reward": "Reward", "mean_rel_strength": "Strength",
                   "tier2_ratio": "Fraction", "engagement": "Engagement Events"}
    fig.update_layout(**_base_layout(
        title=title, xaxis_title="Episode",
        yaxis_title=y_label_map.get(metric, "Value"),
    ))
    return fig


# ── 2. Loss vs training steps ─────────────────────────────────────
def loss_chart(saved_results, title="Loss vs Training Steps — All Models"):
    fig = go.Figure()
    for label, data in saved_results.items():
        color = MODEL_COLORS.get(label, ACCENT_CYAN)
        losses = data.get("losses", [])
        if not losses:
            continue
        smoothed = _smooth(losses, 50)
        fig.add_trace(go.Scatter(
            x=list(range(len(losses))), y=losses, mode="lines",
            name=f"{label} (raw)", line=dict(color=color, width=0.6), opacity=0.15,
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(smoothed))), y=smoothed, mode="lines",
            name=f"{label}", line=dict(color=color, width=2.5),
        ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Optimizer Step", yaxis_title="Loss"))
    return fig


# ── 3. Moving-average reward with std-dev band ────────────────────
def reward_variance_chart(saved_results, title="Reward Convergence — Mean +/- Std Dev"):
    fig = go.Figure()
    for label, data in saved_results.items():
        color = MODEL_COLORS.get(label, ACCENT_CYAN)
        d = _extract(data)
        rewards = d["rewards"]
        if not rewards:
            continue
        smoothed = _smooth(rewards)
        rstd = _rolling_std(rewards)
        x_sm = list(range(len(smoothed)))
        x_std = list(range(19, len(rewards)))   # rolling_std starts at window-1

        # Clip to same length
        min_len = min(len(x_sm), len(x_std), len(rstd))
        x_band = x_std[:min_len]
        sm_band = smoothed[:min_len]
        std_band = rstd[:min_len]

        upper = [s + e for s, e in zip(sm_band, std_band)]
        lower = [s - e for s, e in zip(sm_band, std_band)]

        # Std-dev band
        fig.add_trace(go.Scatter(
            x=x_band + x_band[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=color.replace("ff", "33") if color.startswith("#") else "rgba(100,100,255,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{label} band", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x_sm, y=smoothed, mode="lines",
            name=label, line=dict(color=color, width=2.5),
        ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode", yaxis_title="Reward"))
    return fig


# ── 4. Convergence speed bar chart ───────────────────────────────
def convergence_speed_chart(saved_results, threshold_frac=0.90,
                             title="Convergence Speed (Episodes to 90% of Peak Reward)"):
    labels, speeds = [], []
    for label, data in saved_results.items():
        d = _extract(data)
        rewards = d["rewards"]
        if not rewards:
            continue
        smoothed = _smooth(rewards)
        if not smoothed:
            continue
        target = threshold_frac * max(smoothed)
        idxs = [i for i, v in enumerate(smoothed) if v >= target]
        speed = idxs[0] if idxs else len(rewards)
        labels.append(label)
        speeds.append(speed)

    colors = [MODEL_COLORS.get(l, ACCENT_CYAN) for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=speeds, marker_color=colors,
        text=speeds, textposition="outside",
        marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1,
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Model",
                                      yaxis_title="Episodes"))
    return fig


# ── 5. Training time bar chart ───────────────────────────────────
def training_time_chart(saved_results, title="Training Time Comparison"):
    labels, times = [], []
    for label, data in saved_results.items():
        t = data.get("training_time_s")
        if t is not None:
            labels.append(label)
            times.append(t)
    if not labels:
        return go.Figure()
    colors = [MODEL_COLORS.get(l, ACCENT_CYAN) for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=times, marker_color=colors,
        text=[f"{t:.1f}s" for t in times], textposition="outside",
        marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1,
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Model", yaxis_title="Seconds"))
    return fig


# ── 6. Reward stability (rolling std-dev) ────────────────────────
def stability_chart(saved_results, title="Training Stability — Rolling Reward Std Dev"):
    fig = go.Figure()
    for label, data in saved_results.items():
        color = MODEL_COLORS.get(label, ACCENT_CYAN)
        d = _extract(data)
        rewards = d["rewards"]
        if not rewards:
            continue
        rstd = _rolling_std(rewards)
        x = list(range(19, len(rewards)))
        fig.add_trace(go.Scatter(
            x=x, y=rstd, mode="lines",
            name=label, line=dict(color=color, width=2.5),
        ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode", yaxis_title="Reward Std Dev"))
    return fig


# ── 7. Final relationship strength bar chart ─────────────────────
def relationship_strength_bar(saved_results, title="Final Mean Relationship Strength"):
    labels, vals = [], []
    for label, data in saved_results.items():
        d = _extract(data)
        if d["strengths"]:
            labels.append(label)
            vals.append(round(float(np.mean(d["strengths"][-20:])), 4))
    if not labels:
        return go.Figure()
    colors = [MODEL_COLORS.get(l, ACCENT_CYAN) for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
        marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1,
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Model", yaxis_title="Mean Strength [0-1]"))
    return fig


# ── 8. Engagement vs episodes ─────────────────────────────────────
def engagement_chart(saved_results, title="User Engagement per Episode — All Models"):
    fig = go.Figure()
    for label, data in saved_results.items():
        color = MODEL_COLORS.get(label, ACCENT_CYAN)
        d = _extract(data)
        eng = d["engagements"]
        if not eng or all(v == 0 for v in eng):
            continue
        smoothed = _smooth(eng)
        fig.add_trace(go.Scatter(
            x=list(range(len(eng))), y=eng, mode="lines",
            name=f"{label} (raw)", line=dict(color=color, width=0.8), opacity=0.2,
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(smoothed))), y=smoothed, mode="lines",
            name=label, line=dict(color=color, width=2.5),
        ))
    fig.update_layout(**_base_layout(title=title, xaxis_title="Episode",
                                      yaxis_title="Engagement Events"))
    return fig
