"""
frontend/components/analytics_panel.py
RL Analytics Panel — full training analysis for Q-Learning, DQN, and PPO.

Sections:
  Tab 1 — Live Session   : reward, relationship, epsilon, tier-2 for current session
  Tab 2 — Offline Comparison : all 3-model plots loaded from data/synthetic/
  Tab 3 — Regenerate Plots   : run training and regenerate all static PNG plots
"""

import os
import subprocess
import sys
import streamlit as st

from frontend.charts.analytics import (
    comparison_chart, loss_chart, reward_variance_chart,
    convergence_speed_chart, training_time_chart,
    stability_chart, relationship_strength_bar, engagement_chart,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "data", "synthetic", "plots")
VENV_PYTHON  = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")


def _venv_python():
    """Return the best available Python interpreter (venv > current)."""
    if os.path.exists(VENV_PYTHON):
        return VENV_PYTHON
    return sys.executable


def render_analytics_panel(bridge):
    st.markdown("""
    <div class="section-header">📈 RL Analytics & Training Progress</div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "📊 All-Models Comparison",
        "🔁 Regenerate Plots",
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — ALL-MODELS COMPARISON
    # ════════════════════════════════════════════════════════════
    with tab1:
        saved_results = bridge.load_saved_results()
        if not saved_results:
            st.warning(
                "No offline training results found in **data/synthetic/**.\n\n"
                "Go to the **Regenerate Plots** tab and click **Run All Training** to generate them."
            )
        else:
            found = list(saved_results.keys())
            st.caption(f"Loaded results for: **{', '.join(found)}**")

            # ── Row 1: Reward convergence + Reward variance
            c1, c2 = st.columns(2)
            with c1:
                metric_sel = st.selectbox(
                    "Metric for overlay chart",
                    ["Cumulative Reward", "Mean Relationship Strength",
                     "Tier-2 Feed Ratio", "User Engagement"],
                    key="metric_sel",
                )
                metric_map = {
                    "Cumulative Reward":        "total_reward",
                    "Mean Relationship Strength": "mean_rel_strength",
                    "Tier-2 Feed Ratio":        "tier2_ratio",
                    "User Engagement":          "engagement",
                }
                st.plotly_chart(
                    comparison_chart(saved_results, metric=metric_map[metric_sel],
                                     title=f"All Models — {metric_sel}"),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with c2:
                st.plotly_chart(
                    reward_variance_chart(saved_results,
                                          title="Reward Convergence — Mean +/- Std Dev"),
                    use_container_width=True, config={"displayModeBar": False},
                )

            st.markdown("---")

            # ── Row 2: Loss + Engagement
            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(
                    loss_chart(saved_results, title="Loss vs Training Steps"),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with c4:
                st.plotly_chart(
                    engagement_chart(saved_results, title="User Engagement per Episode"),
                    use_container_width=True, config={"displayModeBar": False},
                )

            st.markdown("---")

            # ── Row 3: Convergence speed + Training time
            c5, c6 = st.columns(2)
            with c5:
                st.plotly_chart(
                    convergence_speed_chart(saved_results,
                                            title="Convergence Speed (Episodes to 90% of Peak)"),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with c6:
                st.plotly_chart(
                    training_time_chart(saved_results, title="Training Time Comparison"),
                    use_container_width=True, config={"displayModeBar": False},
                )

            st.markdown("---")

            # ── Row 4: Stability + Relationship strength
            c7, c8 = st.columns(2)
            with c7:
                st.plotly_chart(
                    stability_chart(saved_results,
                                    title="Training Stability — Rolling Reward Std Dev"),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with c8:
                st.plotly_chart(
                    relationship_strength_bar(saved_results,
                                              title="Final Mean Relationship Strength"),
                    use_container_width=True, config={"displayModeBar": False},
                )

    # ════════════════════════════════════════════════════════════
    # TAB 2 — REGENERATE PLOTS
    # ════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div style="color:#9ca3af; font-size:0.9rem; margin-bottom:1rem;">
        Re-train any model from scratch and regenerate all static PNG plots saved to
        <code>data/synthetic/plots/</code>. Choose the number of episodes before training.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            num_eps = st.number_input(
                "Episodes per model", min_value=50, max_value=5000,
                value=500, step=50, key="regen_episodes",
            )

        st.markdown("#### Train Individual Models")
        r1, r2, r3 = st.columns(3)

        def _run_training(script_rel, episodes):
            """Run a training script with the given episode count via venv Python."""
            script = os.path.join(PROJECT_ROOT, script_rel)
            py = _venv_python()
            env_vars = os.environ.copy()
            env_vars["FAMILYLINK_NUM_EPISODES"] = str(int(episodes))
            result = subprocess.run(
                [py, script],
                capture_output=True, text=True,
                cwd=PROJECT_ROOT, env=env_vars,
            )
            return result

        with r1:
            if st.button("Train Q-Learning", key="btn_train_q", use_container_width=True):
                with st.spinner(f"Training Q-Learning for {num_eps} episodes..."):
                    res = _run_training("training/train_q.py", num_eps)
                if res.returncode == 0:
                    st.success("Q-Learning training complete!")
                else:
                    st.error(f"Error:\n{res.stderr[-800:]}")

        with r2:
            if st.button("Train DQN", key="btn_train_dqn", use_container_width=True):
                with st.spinner(f"Training DQN for {num_eps} episodes..."):
                    res = _run_training("training/train_dqn.py", num_eps)
                if res.returncode == 0:
                    st.success("DQN training complete!")
                else:
                    st.error(f"Error:\n{res.stderr[-800:]}")

        with r3:
            if st.button("Train PPO", key="btn_train_ppo", use_container_width=True):
                with st.spinner(f"Training PPO for {num_eps} episodes..."):
                    res = _run_training("training/train_ppo.py", num_eps)
                if res.returncode == 0:
                    st.success("PPO training complete!")
                else:
                    st.error(f"Error:\n{res.stderr[-800:]}")

        st.markdown("---")

        col_full, col_plots = st.columns(2)

        with col_full:
            st.markdown("#### Train All 3 Models")
            if st.button("Run All Training", key="btn_train_all", use_container_width=True):
                models = [
                    ("training/train_q.py",   "Q-Learning"),
                    ("training/train_dqn.py",  "DQN"),
                    ("training/train_ppo.py",  "PPO"),
                ]
                all_ok = True
                for script, name in models:
                    with st.spinner(f"Training {name} for {num_eps} episodes..."):
                        res = _run_training(script, num_eps)
                    if res.returncode == 0:
                        st.success(f"{name} done!")
                    else:
                        st.error(f"{name} failed: {res.stderr[-400:]}")
                        all_ok = False
                if all_ok:
                    st.balloons()

        with col_plots:
            st.markdown("#### Regenerate PNG Plot Files")
            st.caption(f"Saved to: `data/synthetic/plots/`")
            if st.button("Generate All Plots", key="btn_gen_plots", use_container_width=True):
                with st.spinner("Generating plots..."):
                    py = _venv_python()
                    result = subprocess.run(
                        [py, os.path.join(PROJECT_ROOT, "evaluation", "full_plots.py")],
                        capture_output=True, text=True, cwd=PROJECT_ROOT,
                    )
                if result.returncode == 0:
                    st.success("All plots generated!")
                    # Show them inline
                    _show_saved_plots()
                else:
                    st.error(f"Plot generation failed:\n{result.stderr[-800:]}")

        st.markdown("---")
        st.markdown("#### View Saved PNG Plots")
        st.caption(f"Location: `{PLOTS_DIR}`")
        if st.button("Refresh / Show Plots", key="btn_show_plots", use_container_width=True):
            _show_saved_plots()


def _show_saved_plots():
    """Display all saved PNG plots from the plots directory inline."""
    if not os.path.isdir(PLOTS_DIR):
        st.warning("No plots directory found. Run 'Generate All Plots' first.")
        return

    png_files = sorted([
        f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")
    ])

    if not png_files:
        st.warning("No PNG files found. Run 'Generate All Plots' first.")
        return

    # Group into per-model vs comparison
    per_model = [f for f in png_files if not f.startswith("compare") and f != "full_dashboard.png" and "subloss" not in f]
    comparisons = [f for f in png_files if f.startswith("compare")]
    extras = [f for f in png_files if f not in per_model and f not in comparisons]

    if per_model:
        st.markdown("**Per-Model Plots**")
        cols = st.columns(2)
        for i, fname in enumerate(per_model):
            path = os.path.join(PLOTS_DIR, fname)
            label = fname.replace("_", " ").replace(".png", "").title()
            cols[i % 2].image(path, caption=label, use_container_width=True)

    if comparisons:
        st.markdown("**Cross-Model Comparison Plots**")
        cols = st.columns(2)
        for i, fname in enumerate(comparisons):
            path = os.path.join(PLOTS_DIR, fname)
            label = fname.replace("_", " ").replace(".png", "").title()
            cols[i % 2].image(path, caption=label, use_container_width=True)

    if extras:
        st.markdown("**Summary Dashboards**")
        for fname in extras:
            path = os.path.join(PLOTS_DIR, fname)
            label = fname.replace("_", " ").replace(".png", "").title()
            st.image(path, caption=label, use_container_width=True)
