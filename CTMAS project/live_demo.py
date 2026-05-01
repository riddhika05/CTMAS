import os
import time
import random
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
import lime
import lime.lime_tabular
import warnings

warnings.filterwarnings('ignore')

# ANSI colors for terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_system():
    print(f"{YELLOW}Initializing CTMAS Live Threat Detection System...{RESET}")

    # Load model
    model_path = "outputs/xgboost_model.json"
    if not os.path.exists(model_path):
        print(f"{RED}Error: Model not found at {model_path}. Please run ctmas_pipeline.py first.{RESET}")
        return None, None, None, None, None

    model = XGBClassifier()
    model.load_model(model_path)

    # Load some data for simulation
    print(f"{YELLOW}Loading sensor stream data...{RESET}")
    df = pd.read_csv("archive3/clean_merged.csv", index_col=0, parse_dates=True)

    # Separate normal and attack to ensure we show a good mix in the demo
    normal_df = df[df['label'] == 0].drop(columns=['label'])
    attack_df = df[df['label'] == 1].drop(columns=['label'])

    # Initialize SHAP explainer
    print(f"{YELLOW}Initializing Explainable AI Engine (SHAP)...{RESET}")
    shap_explainer = shap.TreeExplainer(model)

    # Initialize LIME explainer with normal data as training background
    print(f"{YELLOW}Initializing LIME Explainer (using training data background)...{RESET}")
    lime_bg = normal_df.sample(min(10000, len(normal_df)), random_state=42)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_bg.values,
        feature_names=list(normal_df.columns),
        class_names=["Normal", "Attack"],
        mode="classification",
        random_state=42,
    )

    return model, shap_explainer, lime_explainer, normal_df, attack_df

def run_live_demo(model, shap_explainer, lime_explainer, normal_df, attack_df):
    clear_screen()
    print("=" * 70)
    print(f"{BOLD}CTMAS LIVE DASHBOARD - SECURE WATER TREATMENT MONITORING{RESET}")
    print(f"{BOLD}Dual XAI Engine: SHAP + LIME{RESET}")
    print("=" * 70)
    print("Streaming live sensor data from CPS network...")
    print("Press Ctrl+C to stop the simulation.\n")

    feature_names = list(normal_df.columns)

    try:
        while True:
            # 80% chance normal, 20% chance attack for dramatic effect
            is_attack_sim = random.random() > 0.8

            if is_attack_sim and len(attack_df) > 0:
                row = attack_df.sample(1)
            else:
                row = normal_df.sample(1)

            timestamp = row.index[0]

            # Predict
            prob = model.predict_proba(row)[0, 1]
            pred = 1 if prob >= 0.3 else 0

            if pred == 0:
                print(f"[{timestamp}] Status: {GREEN}NORMAL{RESET} (Threat Prob: {prob:.4f})")
                time.sleep(1.0)
            else:
                print(f"\n[{timestamp}] Status: {RED}{BOLD}🚨 CRITICAL ALERT - CYBER ATTACK DETECTED 🚨{RESET}")
                print(f"[{timestamp}] {RED}Threat Probability: {prob:.4f}{RESET}")

                # Generate real-time SHAP explanation
                print(f"[{timestamp}] {YELLOW}SHAP Engine analyzing...{RESET}", end="")
                shap_values = shap_explainer.shap_values(row)
                instance_shap = shap_values[0]
                abs_shap = np.abs(instance_shap)
                shap_top_idx = np.argsort(abs_shap)[::-1][:3]
                shap_feats = [feature_names[j] for j in shap_top_idx]
                shap_vals = [float(row.iloc[0, j]) for j in shap_top_idx]
                print(f" {GREEN}done{RESET}")

                # Generate real-time LIME explanation
                print(f"[{timestamp}] {YELLOW}LIME Engine analyzing...{RESET}", end="")
                lime_exp = lime_explainer.explain_instance(
                    row.values[0],
                    model.predict_proba,
                    num_features=3,
                    num_samples=500,
                    labels=(1,),
                )
                lime_weights = lime_exp.as_list(label=1)[:3]
                lime_feats = []
                for feat_expr, w in lime_weights:
                    matched = [f for f in feature_names if f in feat_expr]
                    lime_feats.append(matched[0] if matched else feat_expr.split(" ")[0])
                print(f" {GREEN}done{RESET}")

                # Agreement check
                agree = shap_feats[0] == lime_feats[0]
                agree_str = f"{GREEN}✅ AGREE{RESET}" if agree else f"{RED}⚠️ DIFFER{RESET}"

                print(f"[{timestamp}] {BOLD}SHAP Top-3:{RESET} {', '.join(shap_feats)}")
                print(f"[{timestamp}] {BOLD}LIME Top-3:{RESET} {', '.join(lime_feats)}")
                print(f"[{timestamp}] {BOLD}Agreement:{RESET}  {agree_str} (Top-1: SHAP={shap_feats[0]}, LIME={lime_feats[0]})")
                print(f"[{timestamp}] {BOLD}Sensor Details (SHAP):{RESET}")
                for f, v in zip(shap_feats, shap_vals):
                    print(f"           - {f}: {v:.3f}")
                print()
                time.sleep(2.5)  # Pause longer on an attack

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Live monitoring terminated by user.{RESET}")

if __name__ == "__main__":
    model, shap_explainer, lime_explainer, normal_df, attack_df = load_system()
    if model is not None:
        time.sleep(1)
        run_live_demo(model, shap_explainer, lime_explainer, normal_df, attack_df)
