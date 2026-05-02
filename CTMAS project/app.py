import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
from datetime import datetime, timezone
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CTMAS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# FIX 7 — CSS OVERHAUL
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .ctmas-card {
        background: #0e1117;
        border: 1px solid #2d2d3a;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .section-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] {
        background-color: #0a0a12;
        border-right: 1px solid #1e1e2e;
    }

    /* FIX 3 — Attack alert with pulse animation */
    .attack-alert-v2 {
        background: #0d0d0d;
        border-left: 3px solid #ff4b4b;
        color: #ffffff;
        padding: 20px 24px;
        border-radius: 10px;
        animation: pulse-red 1.5s ease-in-out infinite;
    }
    .attack-alert-v2 h2 { margin: 0 0 6px 0; font-size: 1.4em; }
    .attack-alert-v2 p  { margin: 0; color: #ccc; font-size: 0.95em; }
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0 0 rgba(255,75,75,0.8); }
        70%  { box-shadow: 0 0 0 16px rgba(255,75,75,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,75,75,0); }
    }

    .normal-status-v2 {
        background: #0d0d0d;
        border-left: 3px solid #00cc96;
        color: #ffffff;
        padding: 20px 24px;
        border-radius: 10px;
    }
    .normal-status-v2 h2 { margin: 0 0 6px 0; font-size: 1.4em; }
    .normal-status-v2 p  { margin: 0; color: #ccc; font-size: 0.95em; }

    .sidebar-stat { font-size: 0.85em; color: #aaa; margin: 2px 0; }
    .dot-green { display:inline-block; width:8px; height:8px; border-radius:50%; background:#00cc96; margin-right:6px; }
    .dot-grey  { display:inline-block; width:8px; height:8px; border-radius:50%; background:#555;    margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD ASSETS (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    model_path = "outputs/xgboost_model.json"
    if not os.path.exists(model_path):
        return None, None, None, None, None
    model = XGBClassifier()
    model.load_model(model_path)
    df = pd.read_csv("archive3/clean_merged.csv", index_col=0, parse_dates=True)
    normal_df = df[df["label"] == 0].drop(columns=["label"])
    attack_df = df[df["label"] == 1].drop(columns=["label"])
    explainer = shap.TreeExplainer(model)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=normal_df.sample(min(10000, len(normal_df)), random_state=42).values,
        feature_names=list(normal_df.columns),
        class_names=["Normal", "Attack"],
        mode="classification",
        random_state=42,
    )
    return model, explainer, lime_explainer, normal_df, attack_df


# =============================================================================
# FIX 1 — Load top-8 sensors by SHAP importance
# =============================================================================
@st.cache_data
def get_top_sensors(fallback_features, k=8):
    """Return fixed list of top-k sensor names ranked by SHAP importance."""
    path = "outputs/shap_summary.csv"
    if os.path.exists(path):
        shap_df = pd.read_csv(path)
        return shap_df["feature"].head(k).tolist()
    return list(fallback_features[:k])


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
    st.sidebar.title("CTMAS")
    st.sidebar.markdown("**Cyber-Physical Threat Monitoring & Analysis System**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
        "🔴 Live Monitoring Demo",
        "📊 System Results & Evaluation",
        "ℹ️ About System",
    ])

    # FIX 5 — init session state
    if "attack_log" not in st.session_state:
        st.session_state["attack_log"] = []
    if "prev_sensor_vals" not in st.session_state:
        st.session_state["prev_sensor_vals"] = {}

    # FIX 8 — Sidebar live stats
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="section-label">System Stats</p>', unsafe_allow_html=True)
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_attacks_session = len(st.session_state["attack_log"])
    is_running = st.session_state.get("running", False)
    dot_cls = "dot-green" if is_running else "dot-grey"
    status_txt = "STREAMING" if is_running else "IDLE"
    st.sidebar.markdown(
        f'<p class="sidebar-stat">🕐 {utc_now}</p>'
        f'<p class="sidebar-stat">🚨 Attacks this session: <b>{n_attacks_session}</b></p>'
        f'<p class="sidebar-stat"><span class="{dot_cls}"></span>{status_txt}</p>',
        unsafe_allow_html=True,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    model, shap_explainer, lime_explainer, normal_df, attack_df = load_system()
    if model is None:
        st.error("Model not found! Run the pipeline first.")
        return

    feature_names = list(normal_df.columns)
    top_sensors = get_top_sensors(feature_names)

    # =====================================================================
    # PAGE 1: LIVE MONITORING
    # =====================================================================
    if page == "🔴 Live Monitoring Demo":
        st.title("🔴 Live CPS Monitoring Dashboard")
        st.markdown("Simulating real-time sensor stream from the SWaT Water Treatment Plant.")

        ctrl_col, main_col = st.columns([1, 4])

        with ctrl_col:
            st.markdown('<p class="section-label">Controls</p>', unsafe_allow_html=True)
            start_btn = st.button("▶️ Start Live Feed", use_container_width=True, type="primary")
            stop_btn = st.button("⏹️ Stop Feed", use_container_width=True)
            speed = st.slider("Simulation Speed (s)", 0.5, 3.0, 1.5)
            attack_prob = st.slider("Attack Injection %", 0.0, 1.0, 0.2,
                                    help="Probability of injecting an attack row")

        # FIX 2 — All placeholders defined before the loop
        with main_col:
            status_placeholder = st.empty()
            prob_placeholder = st.empty()
            st.markdown("### 📡 Key Sensor Readings (Top 8 by SHAP Importance)")
            sensors_placeholder = st.empty()
            st.markdown("### 🧠 XAI Threat Reasoning (SHAP + LIME)")
            xai_placeholder = st.empty()

        # FIX 5 — Attack log section (outside loop)
        st.markdown("---")
        st.markdown("### 🗂 Session Attack Log")
        log_clear_col, _ = st.columns([1, 5])
        with log_clear_col:
            if st.button("🗑 Clear Log", use_container_width=True):
                st.session_state["attack_log"] = []
        attack_log_placeholder = st.empty()
        # Render existing log
        if st.session_state["attack_log"]:
            attack_log_placeholder.dataframe(
                pd.DataFrame(st.session_state["attack_log"]),
                use_container_width=True, hide_index=True,
            )

        if start_btn:
            st.session_state["running"] = True
        if stop_btn:
            st.session_state["running"] = False

        # ── Live loop ────────────────────────────────────────────────────────
        if st.session_state.get("running", False):
            while st.session_state.get("running", False):
                # Sample row
                if random.random() < attack_prob and len(attack_df) > 0:
                    row = attack_df.sample(1)
                else:
                    row = normal_df.sample(1)
                timestamp = row.index[0]

                # Predict (ML logic unchanged)
                prob = float(model.predict_proba(row)[0, 1])
                pred = 1 if prob >= 0.3 else 0

                # Progress bar
                prob_placeholder.progress(prob, text=f"Threat Probability: {prob:.4f}")

                # FIX 1 — Fixed sensors with delta
                with sensors_placeholder.container():
                    r1 = st.columns(4)
                    r2 = st.columns(4)
                    cols_flat = r1 + r2
                    prev = st.session_state["prev_sensor_vals"]
                    new_prev = {}
                    for idx, sensor in enumerate(top_sensors):
                        val = float(row[sensor].values[0])
                        delta = round(val - prev[sensor], 2) if sensor in prev else None
                        cols_flat[idx].metric(label=sensor, value=f"{val:.2f}",
                                              delta=f"{delta}" if delta is not None else None)
                        new_prev[sensor] = val
                    st.session_state["prev_sensor_vals"] = new_prev

                # ── Normal path ──────────────────────────────────────────────
                if pred == 0:
                    status_placeholder.markdown(
                        f'<div class="normal-status-v2">'
                        f'<h2>✅ NORMAL OPERATION</h2>'
                        f'<p>Timestamp: {timestamp} · Threat Prob: {prob:.4f}</p></div>',
                        unsafe_allow_html=True,
                    )
                    with xai_placeholder.container():
                        st.info("System operating within normal physical boundaries. No anomalies.")
                    time.sleep(speed)

                # ── Attack path ──────────────────────────────────────────────
                else:
                    # FIX 3 — Pulse-red alert card
                    status_placeholder.markdown(
                        f'<div class="attack-alert-v2">'
                        f'<h2>🚨 CYBER-PHYSICAL ATTACK DETECTED</h2>'
                        f'<p>Timestamp: {timestamp} · Threat Prob: {prob:.4f}</p></div>',
                        unsafe_allow_html=True,
                    )

                    # FIX 4 — XAI dual pane inside placeholder container
                    with xai_placeholder.container():
                        # SHAP (ML logic unchanged)
                        shap_values = shap_explainer.shap_values(row)
                        instance_shap = shap_values[0]
                        abs_shap = np.abs(instance_shap)
                        shap_top_idx = np.argsort(abs_shap)[::-1][:3]
                        shap_feats = [feature_names[j] for j in shap_top_idx]
                        shap_vals = [float(instance_shap[j]) for j in shap_top_idx]

                        # LIME (ML logic unchanged)
                        lime_exp = lime_explainer.explain_instance(
                            row.values[0], model.predict_proba,
                            num_features=3, num_samples=500, labels=(1,),
                        )
                        lime_weights = lime_exp.as_list(label=1)[:3]
                        lime_feats, lime_vals = [], []
                        for feat_expr, w in lime_weights:
                            matched = [f for f in feature_names if f in feat_expr]
                            lime_feats.append(matched[0] if matched else feat_expr.split(" ")[0])
                            lime_vals.append(w)

                        agree = shap_feats[0] == lime_feats[0]

                        # Charts side-by-side
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**SHAP — Feature Contributions**")
                            fig1, ax1 = plt.subplots(figsize=(6, 2.5))
                            colors = ["#ff4b4b" if s > 0 else "#00cc96" for s in shap_vals]
                            ax1.barh(shap_feats[::-1], shap_vals[::-1], color=colors[::-1])
                            ax1.set_xlabel("SHAP Impact")
                            ax1.set_facecolor("#0e1117")
                            fig1.patch.set_facecolor("#0e1117")
                            ax1.tick_params(colors="#ccc"); ax1.xaxis.label.set_color("#ccc")
                            for spine in ax1.spines.values(): spine.set_color("#2d2d3a")
                            plt.tight_layout()
                            st.pyplot(fig1)
                            plt.close(fig1)
                        with c2:
                            st.markdown("**LIME — Local Surrogate Weights**")
                            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
                            colors2 = ["#ff4b4b" if w > 0 else "#00cc96" for w in lime_vals]
                            ax2.barh(lime_feats[::-1], lime_vals[::-1], color=colors2[::-1])
                            ax2.set_xlabel("LIME Weight")
                            ax2.set_facecolor("#0e1117")
                            fig2.patch.set_facecolor("#0e1117")
                            ax2.tick_params(colors="#ccc"); ax2.xaxis.label.set_color("#ccc")
                            for spine in ax2.spines.values(): spine.set_color("#2d2d3a")
                            plt.tight_layout()
                            st.pyplot(fig2)
                            plt.close(fig2)

                        # Agreement line
                        if agree:
                            st.success(f"✅ XAI Consensus: Both SHAP and LIME identify "
                                       f"**{shap_feats[0]}** as the primary manipulated sensor.")
                        else:
                            st.warning(f"⚠️ XAI Conflict: SHAP flags **{shap_feats[0]}**, "
                                       f"LIME flags **{lime_feats[0]}**. Manual review recommended.")

                    # FIX 5 — Append to attack log
                    st.session_state["attack_log"].append({
                        "Timestamp": str(timestamp),
                        "Threat Prob": f"{prob:.4f}",
                        "Top Sensor (SHAP)": shap_feats[0],
                        "Top Sensor (LIME)": lime_feats[0],
                        "Agreement": "✅ AGREE" if agree else "⚠️ DIFFER",
                    })
                    attack_log_placeholder.dataframe(
                        pd.DataFrame(st.session_state["attack_log"]),
                        use_container_width=True, hide_index=True,
                    )

                    time.sleep(speed * 3)

    # =====================================================================
    # PAGE 2: SYSTEM RESULTS — FIX 6 (tabbed layout)
    # =====================================================================
    elif page == "📊 System Results & Evaluation":
        st.title("📊 System Evaluation Results")
        st.markdown("Performance metrics from the **XGBoost + SHAP + LIME** pipeline on the SWaT dataset.")

        tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🧠 XAI Analysis", "🔬 Attack Clusters"])

        # ── Tab 1: Model Performance ─────────────────────────────────────────
        with tab1:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", "98.19%")
            m2.metric("Precision", "67.73%", help="When it predicts attack, it's right 67% of the time")
            m3.metric("Recall (Detection)", "99.92%", help="Caught 99.92% of all actual attacks")
            m4.metric("F1-Score", "80.74%")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Confusion Matrix")
                if os.path.exists("outputs/confusion_matrix.png"):
                    st.image("outputs/confusion_matrix.png", use_container_width=True)
                else:
                    st.warning("Not found. Run pipeline first.")
            with c2:
                st.markdown("#### Global SHAP Importance")
                if os.path.exists("outputs/shap_global_bar.png"):
                    st.image("outputs/shap_global_bar.png", use_container_width=True)
                else:
                    st.warning("Not found. Run pipeline first.")

            # Optional plots (ROC, PR, threshold)
            opt_col1, opt_col2 = st.columns(2)
            for col, fname, title in [
                (opt_col1, "outputs/roc_curve.png", "ROC Curve"),
                (opt_col2, "outputs/pr_curve.png", "Precision-Recall Curve"),
            ]:
                with col:
                    if os.path.exists(fname):
                        st.markdown(f"#### {title}")
                        st.image(fname, use_container_width=True)

            if os.path.exists("outputs/threshold_analysis.png"):
                st.markdown("#### Threshold Analysis")
                st.image("outputs/threshold_analysis.png", use_container_width=True)

        # ── Tab 2: XAI Analysis ──────────────────────────────────────────────
        with tab2:
            st.markdown("#### SHAP Beeswarm Plot")
            if os.path.exists("outputs/shap_beeswarm.png"):
                st.image("outputs/shap_beeswarm.png", use_container_width=True)
            else:
                st.warning("Not found. Run pipeline first.")

            st.markdown("---")
            st.markdown("#### 🍋 LIME Feature Importance")
            if os.path.exists("outputs/lime_summary.csv"):
                lime_df = pd.read_csv("outputs/lime_summary.csv")
                top_lime = lime_df.head(20)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(top_lime["feature"][::-1], top_lime["mean_abs_lime_weight"][::-1],
                        color="#e67e22", alpha=0.85)
                ax.set_xlabel("Mean |LIME Weight|", fontsize=12)
                ax.set_title("Top-20 Features — Global LIME Importance", fontsize=13)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("LIME results not found. Run pipeline with LIME enabled.")

            st.markdown("---")
            st.markdown("#### 🤝 SHAP vs LIME Agreement")
            if os.path.exists("outputs/xai_agreement_summary.txt"):
                with open("outputs/xai_agreement_summary.txt", "r") as f:
                    summary_text = f.read()
                lines = summary_text.strip().split("\n")
                top1_line = [l for l in lines if "Top-1" in l]
                jaccard_line = [l for l in lines if "Jaccard" in l]

                col_a, col_b = st.columns(2)
                if top1_line:
                    pct = top1_line[0].split(":")[1].strip().replace("%", "")
                    col_a.metric("Top-1 Feature Agreement", f"{pct}%",
                                 help="% of attacks where SHAP & LIME agree on #1 sensor")
                if jaccard_line:
                    jac = jaccard_line[0].split(":")[1].strip().replace("%", "")
                    col_b.metric("Top-3 Jaccard Overlap", f"{jac}%",
                                 help="Average overlap between top-3 features")

                finding_lines = [l for l in lines if "Finding" in l]
                if finding_lines:
                    st.info("**Key Finding:** " + finding_lines[0].split("Finding:")[1].strip())
            else:
                st.warning("Agreement analysis not found. Run pipeline first.")

            if os.path.exists("outputs/xai_agreement.csv"):
                with st.expander("📋 Per-Instance Agreement Table"):
                    agree_df = pd.read_csv("outputs/xai_agreement.csv")
                    st.dataframe(agree_df.head(100), use_container_width=True, hide_index=True)

        # ── Tab 3: Attack Clusters ───────────────────────────────────────────
        with tab3:
            if os.path.exists("outputs/shap_attack_clusters.png"):
                st.image("outputs/shap_attack_clusters.png", use_container_width=True,
                         caption="PCA projection of SHAP vectors, colored by KMeans cluster")
            else:
                st.warning("Clustering plot not found. Run pipeline first.")

            if os.path.exists("outputs/attack_cluster_summary.csv"):
                summary_df = pd.read_csv("outputs/attack_cluster_summary.csv")
                sil = summary_df["silhouette_score"].iloc[0] if "silhouette_score" in summary_df.columns else None
                n_clusters = len(summary_df)
                total_attacks = int(summary_df["size"].sum())

                h1, h2, h3 = st.columns(3)
                h1.metric("Clusters Found", n_clusters)
                h2.metric("Total Attacks Clustered", f"{total_attacks:,}")
                if sil is not None:
                    h3.metric("Silhouette Score", f"{sil:.4f}",
                              help="Cluster quality (higher = better, max 1.0)")

                st.markdown("#### Cluster Summary")
                display_df = summary_df.copy()
                display_df.columns = [
                    "Cluster", "Size", "% of Attacks", "Top Stat Sensor",
                    "Top SHAP Sensor", "Agreement", "Label", "Silhouette",
                ]

                def _hl(val):
                    if val is True or val == "True":
                        return "background-color: #00cc9644"
                    return "background-color: #ff4b4b44"

                st.dataframe(
                    display_df.style.applymap(_hl, subset=["Agreement"]),
                    use_container_width=True, hide_index=True,
                )

                st.markdown("#### Cluster Details")
                cols = st.columns(min(n_clusters, 4))
                for i, row in summary_df.iterrows():
                    with cols[i % min(n_clusters, 4)]:
                        pct = row["pct_of_attacks"]
                        warn = " ⚠️" if pct < 5.0 else ""
                        icon = "✅" if row["agreement"] else "⚠️"
                        border = "#00cc96" if row["agreement"] else "#ff4b4b"
                        st.markdown(f"""
                        <div class="ctmas-card" style="border-left:4px solid {border};">
                            <h4 style="margin:0;">Cluster {row['cluster']}</h4>
                            <p style="font-size:0.85em;color:#888;margin:5px 0;">{row['label']}</p>
                            <p style="margin:3px 0;"><b>{row['size']:,}</b> attacks ({pct:.1f}%){warn}</p>
                            <p style="margin:3px 0;">Stat: <code>{row['top_stat_sensor']}</code> | SHAP: <code>{row['top_shap_sensor']}</code> {icon}</p>
                        </div>""", unsafe_allow_html=True)
            else:
                st.warning("Cluster summary not found. Run pipeline first.")

    # =====================================================================
    # PAGE 3: ABOUT
    # =====================================================================
    elif page == "ℹ️ About System":
        st.title("ℹ️ About CTMAS")
        st.markdown("""
        ### Cyber-Physical Threat Monitoring & Analysis System

        **How it works:**
        1. **Detection (XGBoost):** Trains on the SWaT (Secure Water Treatment) dataset — 51 sensors and actuators. When live data violates learned physical constraints, it flags the row as an attack.
        2. **Explainability (SHAP):** Uses Game Theory (SHapley Additive exPlanations) to isolate which sensors caused the anomaly.
        3. **Explainability (LIME):** Creates local surrogate models by perturbing inputs and observing prediction changes — a complementary view.
        4. **XAI Agreement Score:** Compares top features from SHAP and LIME. High agreement validates explanation trustworthiness.
        5. **Attack Clustering (SHAP + KMeans + PCA):** Clusters attacks using KMeans on SHAP vectors, cross-references statistical deviation with SHAP importance per cluster to auto-generate labels (e.g., "LIT101-dominant attack"). PCA reduces to 2D for visualization.

        **Built with:** Python · XGBoost · SHAP · LIME · scikit-learn · Streamlit
        """)


if __name__ == "__main__":
    main()
