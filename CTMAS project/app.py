import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import random
from xgboost import XGBClassifier
import shap
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD ASSETS (Cached for speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    model_path = "outputs/xgboost_model.json"
    if not os.path.exists(model_path):
        return None, None, None, None
        
    model = XGBClassifier()
    model.load_model(model_path)
    
    df = pd.read_csv("archive3/clean_merged.csv", index_col=0, parse_dates=True)
    
    normal_df = df[df['label'] == 0].drop(columns=['label'])
    attack_df = df[df['label'] == 1].drop(columns=['label'])
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, normal_df, attack_df

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.sidebar.title("CTMAS")
    st.sidebar.markdown("**Cyber-Physical Threat Monitoring & Analysis System**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", ["🔴 Live Monitoring Demo", "📊 System Results & Evaluation", "ℹ️ About System"])
    
    model, explainer, normal_df, attack_df = load_system()
    
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
            
            st.markdown("### XAI Threat Reasoning")
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
                        st.warning("XAI Engine is analyzing the attack vector...")
                        
                        # SHAP computation
                        shap_values = explainer.shap_values(row)
                        instance_shap = shap_values[0]
                        abs_shap = np.abs(instance_shap)
                        top_idx = np.argsort(abs_shap)[::-1][:3]
                        
                        top_feats = [feature_names[j] for j in top_idx]
                        top_vals = [float(row.iloc[0, j]) for j in top_idx]
                        top_shaps = [float(instance_shap[j]) for j in top_idx]
                        
                        st.error(f"**Root Cause Analysis:** Attack identified due to severe abnormal behavior in **{', '.join(top_feats)}**.")
                        
                        # Plot local explanation
                        fig, ax = plt.subplots(figsize=(8, 3))
                        colors = ['#ff4b4b' if s > 0 else '#00cc96' for s in top_shaps]
                        ax.barh(top_feats[::-1], top_shaps[::-1], color=colors[::-1])
                        ax.set_title("Top Sensor Manipulations (SHAP Values)")
                        ax.set_xlabel("Impact on Attack Probability")
                        st.pyplot(fig)
                        
                    time.sleep(speed * 3) # Pause longer so faculty can read it

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
            st.markdown("### Global Feature Importance")
            if os.path.exists("outputs/shap_global_bar.png"):
                st.image("outputs/shap_global_bar.png", use_container_width=True)
            else:
                st.warning("Image not found. Run pipeline first.")
                
        st.markdown("### Deep Dive: SHAP Beeswarm Plot")
        if os.path.exists("outputs/shap_beeswarm.png"):
            st.image("outputs/shap_beeswarm.png", use_container_width=True)
        else:
            st.warning("Image not found. Run pipeline first.")

    # =========================================================================
    # TAB 3: ABOUT
    # =========================================================================
    elif page == "ℹ️ About System":
        st.title("ℹ️ About CTMAS")
        st.markdown("""
        ### Cyber-Physical Threat Monitoring & Analysis System
        
        **How it works:**
        1. **Detection (XGBoost):** The system trains on the SWaT (Secure Water Treatment) dataset. It learns the deep, multi-variate physical correlations between 51 sensors and actuators (pumps, valves, flow meters). When live data violates these learned physical constraints, it flags the row as an attack.
        2. **Explainability (SHAP):** Once an attack is flagged, the system uses Game Theory (SHapley Additive exPlanations) to isolate which specific sensors caused the mathematical anomaly. It acts as a "detective," figuring out exactly which machine the hacker manipulated.
        
        **Built with:**
        * Python
        * XGBoost (Extreme Gradient Boosting)
        * SHAP
        * Streamlit
        """)

if __name__ == "__main__":
    main()
