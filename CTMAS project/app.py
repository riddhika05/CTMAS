import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
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
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .attack-alert {
        padding: 20px;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        text-align: center;
        animation: blink 1s linear infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    .normal-status {
        padding: 20px;
        background-color: #00cc96;
        color: white;
        border-radius: 10px;
        text-align: center;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    .agreement-box {
        padding: 25px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #0f3460;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD ASSETS (Cached for speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    model_path = "outputs/xgboost_model.json"
    if not os.path.exists(model_path):
        return None, None, None, None, None

    model = XGBClassifier()
    model.load_model(model_path)

    df = pd.read_csv("archive3/clean_merged.csv", index_col=0, parse_dates=True)

    normal_df = df[df['label'] == 0].drop(columns=['label'])
    attack_df = df[df['label'] == 1].drop(columns=['label'])

    explainer = shap.TreeExplainer(model)

    # Initialise LIME with normal data as training background
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=normal_df.sample(min(10000, len(normal_df)), random_state=42).values,
        feature_names=list(normal_df.columns),
        class_names=["Normal", "Attack"],
        mode="classification",
        random_state=42,
    )

    return model, explainer, lime_explainer, normal_df, attack_df

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.sidebar.title("CTMAS")
    st.sidebar.markdown("**Cyber-Physical Threat Monitoring & Analysis System**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
        "🔴 Live Monitoring Demo",
        "📊 System Results & Evaluation",
        "ℹ️ About System"
    ])

    model, shap_explainer, lime_explainer, normal_df, attack_df = load_system()

    if model is None:
        st.error("Model not found! Please run the pipeline script first to generate the XGBoost model.")
        return

    # =========================================================================
    # TAB 1: LIVE DEMO
    # =========================================================================
    if page == "🔴 Live Monitoring Demo":
        st.title("🔴 Live CPS Monitoring Dashboard")
        st.markdown("Simulating real-time sensor stream from the SWaT Water Treatment Plant.")

        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown("### Controls")
            start_btn = st.button("▶️ Start Live Feed", use_container_width=True, type="primary")
            stop_btn = st.button("⏹️ Stop Feed", use_container_width=True)
            speed = st.slider("Simulation Speed (s)", 0.5, 3.0, 1.5)
            attack_prob = st.slider("Attack Injection Probability", 0.0, 1.0, 0.2, help="Probability of forcing an attack row in the simulation")

        with col2:
            st.markdown("### System Status")
            status_placeholder = st.empty()
            prob_bar = st.progress(0.0)

            st.markdown("### XAI Threat Reasoning (SHAP + LIME)")
            xai_placeholder = st.empty()

            st.markdown("### Active Sensor Readings")
            sensors_placeholder = st.empty()

        if start_btn:
            st.session_state['running'] = True
        if stop_btn:
            st.session_state['running'] = False

        if st.session_state.get('running', False):
            feature_names = list(normal_df.columns)

            # Loop for live simulation
            while st.session_state.get('running', False):
                # Sample row based on injection probability
                if random.random() < attack_prob and len(attack_df) > 0:
                    row = attack_df.sample(1)
                else:
                    row = normal_df.sample(1)

                timestamp = row.index[0]

                # Predict
                prob = float(model.predict_proba(row)[0, 1])
                pred = 1 if prob >= 0.3 else 0

                # Update UI
                prob_bar.progress(prob)

                # Show active sensors
                sensor_cols = st.columns(6)
                disp_sensors = random.sample(feature_names, 12)
                for i, s in enumerate(disp_sensors):
                    val = row[s].values[0]
                    sensor_cols[i%6].metric(label=s, value=f"{val:.2f}")

                if pred == 0:
                    status_placeholder.markdown(f"<div class='normal-status'><h2>✅ NORMAL OPERATION</h2><p>Timestamp: {timestamp} | Threat Prob: {prob:.4f}</p></div>", unsafe_allow_html=True)
                    xai_placeholder.info("System is operating within normal physical boundaries. No abnormalities detected.")
                    time.sleep(speed)
                else:
                    status_placeholder.markdown(f"<div class='attack-alert'><h2>🚨 CRITICAL CYBER ATTACK DETECTED 🚨</h2><p>Timestamp: {timestamp} | Threat Prob: {prob:.4f}</p></div>", unsafe_allow_html=True)

                    with xai_placeholder.container():
                        st.warning("Dual XAI Engine analyzing the attack vector...")

                        # SHAP computation
                        shap_values = shap_explainer.shap_values(row)
                        instance_shap = shap_values[0]
                        abs_shap = np.abs(instance_shap)
                        shap_top_idx = np.argsort(abs_shap)[::-1][:3]
                        shap_feats = [feature_names[j] for j in shap_top_idx]
                        shap_vals = [float(instance_shap[j]) for j in shap_top_idx]

                        # LIME computation (num_samples=500 for speed)
                        lime_exp = lime_explainer.explain_instance(
                            row.values[0],
                            model.predict_proba,
                            num_features=3,
                            num_samples=500,
                            labels=(1,),
                        )
                        lime_weights = lime_exp.as_list(label=1)[:3]
                        lime_feats = []
                        lime_vals = []
                        for feat_expr, w in lime_weights:
                            matched = [f for f in feature_names if f in feat_expr]
                            lime_feats.append(matched[0] if matched else feat_expr.split(" ")[0])
                            lime_vals.append(w)

                        # Agreement check
                        agree = "✅ AGREE" if shap_feats[0] == lime_feats[0] else "⚠️ DIFFER"

                        st.error(f"**Root Cause:** {', '.join(shap_feats)} | Agreement: {agree}")

                        # Side-by-side plots
                        c1, c2 = st.columns(2)
                        with c1:
                            fig1, ax1 = plt.subplots(figsize=(6, 2.5))
                            colors = ['#ff4b4b' if s > 0 else '#00cc96' for s in shap_vals]
                            ax1.barh(shap_feats[::-1], shap_vals[::-1], color=colors[::-1])
                            ax1.set_title("SHAP Values")
                            ax1.set_xlabel("Impact")
                            st.pyplot(fig1)
                            plt.close(fig1)
                        with c2:
                            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
                            colors2 = ['#ff4b4b' if w > 0 else '#00cc96' for w in lime_vals]
                            ax2.barh([lime_feats[i] for i in range(len(lime_feats))][::-1],
                                     lime_vals[::-1], color=colors2[::-1])
                            ax2.set_title("LIME Weights")
                            ax2.set_xlabel("Impact")
                            st.pyplot(fig2)
                            plt.close(fig2)

                    time.sleep(speed * 3)  # Pause longer so faculty can read it

    # =========================================================================
    # TAB 2: SYSTEM RESULTS
    # =========================================================================
    elif page == "📊 System Results & Evaluation":
        st.title("📊 System Evaluation Results")
        st.markdown("Performance metrics based on **80/20 Shuffled Random Split** using the XGBoost Classifier.")

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", "98.19%")
        m2.metric("Precision", "67.73%", help="When it predicts attack, it's right 67% of the time")
        m3.metric("Recall (Detection Rate)", "99.92%", help="It caught 99.92% of all actual attacks!")
        m4.metric("F1-Score", "80.74%")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Confusion Matrix")
            if os.path.exists("outputs/confusion_matrix.png"):
                st.image("outputs/confusion_matrix.png", use_container_width=True)
            else:
                st.warning("Image not found. Run pipeline first.")

        with col2:
            st.markdown("### Global Feature Importance (SHAP)")
            if os.path.exists("outputs/shap_global_bar.png"):
                st.image("outputs/shap_global_bar.png", use_container_width=True)
            else:
                st.warning("Image not found. Run pipeline first.")

        st.markdown("### SHAP Beeswarm Plot")
        if os.path.exists("outputs/shap_beeswarm.png"):
            st.image("outputs/shap_beeswarm.png", use_container_width=True)
        else:
            st.warning("Image not found. Run pipeline first.")

        # ── LIME Feature Importance ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🍋 LIME Feature Importance")
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
            st.warning("LIME results not found. Run the pipeline with LIME enabled first.")

        # ── SHAP vs LIME Agreement ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🤝 SHAP vs LIME Agreement Analysis")
        if os.path.exists("outputs/xai_agreement_summary.txt"):
            with open("outputs/xai_agreement_summary.txt", "r") as f:
                summary_text = f.read()

            # Parse the headline numbers
            lines = summary_text.strip().split("\n")
            top1_line = [l for l in lines if "Top-1" in l]
            jaccard_line = [l for l in lines if "Jaccard" in l]

            col_a, col_b = st.columns(2)
            if top1_line:
                pct = top1_line[0].split(":")[1].strip().replace("%", "")
                col_a.metric("Top-1 Feature Agreement", f"{pct}%",
                             help="Percentage of attacks where SHAP and LIME agree on the #1 most important sensor")
            if jaccard_line:
                jac = jaccard_line[0].split(":")[1].strip().replace("%", "")
                col_b.metric("Top-3 Jaccard Overlap", f"{jac}%",
                             help="Average overlap between the top-3 features identified by SHAP and LIME")

            st.info("**Key Finding:** " + [l for l in lines if "Finding" in l][0].split("Finding:")[1].strip() if any("Finding" in l for l in lines) else summary_text)
        else:
            st.warning("Agreement analysis not found. Run the pipeline first.")

        if os.path.exists("outputs/xai_agreement.csv"):
            with st.expander("📋 Per-Instance Agreement Table"):
                agree_df = pd.read_csv("outputs/xai_agreement.csv")
                st.dataframe(agree_df.head(100), use_container_width=True)

        # ── Attack Clustering ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🔬 Attack Pattern Clustering")
        if os.path.exists("outputs/attack_clusters_pca.png"):
            st.image("outputs/attack_clusters_pca.png", use_container_width=True)
        else:
            st.warning("Clustering plot not found. Run the pipeline first.")

        if os.path.exists("outputs/attack_clusters.csv"):
            cluster_data = pd.read_csv("outputs/attack_clusters.csv")
            cluster_counts = cluster_data["cluster"].value_counts().sort_index()
            st.markdown("#### Cluster Distribution")
            cols = st.columns(len(cluster_counts))
            for i, (cluster_id, count) in enumerate(cluster_counts.items()):
                cols[i].metric(f"Cluster {cluster_id}", f"{count:,} attacks")

    # =========================================================================
    # TAB 3: ABOUT
    # =========================================================================
    elif page == "ℹ️ About System":
        st.title("ℹ️ About CTMAS")
        st.markdown("""
        ### Cyber-Physical Threat Monitoring & Analysis System

        **How it works:**
        1. **Detection (XGBoost):** The system trains on the SWaT (Secure Water Treatment) dataset. It learns the deep, multi-variate physical correlations between 51 sensors and actuators (pumps, valves, flow meters). When live data violates these learned physical constraints, it flags the row as an attack.
        2. **Explainability (SHAP):** Once an attack is flagged, the system uses Game Theory (SHapley Additive exPlanations) to isolate which specific sensors caused the mathematical anomaly.
        3. **Explainability (LIME):** As a second independent XAI method, LIME (Local Interpretable Model-agnostic Explanations) creates local surrogate models by perturbing the input and observing prediction changes. This provides a complementary view of feature importance.
        4. **XAI Agreement Score:** By comparing the top features identified by SHAP and LIME, we compute an agreement score. High agreement validates that the explanations are trustworthy and not artifacts of a single method.
        5. **Attack Clustering (KMeans + PCA):** Detected attacks are grouped into clusters using KMeans, revealing distinct attack patterns (e.g., tank overflow vs. chemical dosing manipulation). PCA reduces the 51-dimensional space to 2D for visualization.

        **Built with:**
        * Python
        * XGBoost (Extreme Gradient Boosting)
        * SHAP (SHapley Additive exPlanations)
        * LIME (Local Interpretable Model-agnostic Explanations)
        * scikit-learn (KMeans, PCA)
        * Streamlit
        """)

if __name__ == "__main__":
    main()
