# Explainable AI–Based Threat Modeling for Trustworthy Cyber–Physical Systems under Intelligent Adversaries

**CTMAS (Cyber-Physical Threat Monitoring & Analysis System)** is a comprehensive framework designed to detect, explain, and categorize cyber-physical attacks in real-time. This project focuses on enhancing the **trustworthiness** of anomaly detection systems by providing multi-modal explanations and automated threat taxonomy, specifically tailored for critical infrastructure like the **Secure Water Treatment (SWaT)** plant.

---

## 🚀 Key Objectives

1.  **Detection under Intelligent Adversaries**: Implementing robust classifiers that can identify stealthy manipulations designed to bypass traditional threshold-based alarms.
2.  **Trustworthy Explainability**: Leveraging independent XAI methods (SHAP & LIME) to validate model reasoning.
3.  **Threat Taxonomy**: Automatically clustering attack patterns to understand adversary strategies (e.g., sensor spoofing vs. actuator sabotage).

---

## 🛠 Model Engineering & Detection Strategy

The core of the system is an **XGBoost (Extreme Gradient Boosting)** binary classifier, optimized for the high-dimensional, imbalanced nature of CPS data.

### 1. Handling Class Imbalance with `scale_pos_weight`
In the SWaT dataset, attacks represent only ~3.8% of the total data. To prevent the model from becoming biased toward "Normal" operation:
- We compute **SPW (Scale Pos Weight)** = `count(negative samples) / count(positive samples)`.
- In our implementation, this ratio is approximately **25.4:1**.
- This forces XGBoost to penalize misclassifications of the minority (Attack) class 25 times more heavily than the majority class.

### 2. Decision Threshold Optimization (0.3)
Unlike standard classifiers that use a 0.5 threshold, CTMAS uses a **0.3 threshold**:
- **Rationale**: In critical infrastructure security, a **False Negative (missing an attack)** is far more dangerous than a False Positive.
- By lowering the threshold, we significantly boost **Recall (Detection Rate)**, ensuring that even low-probability threat signals are flagged for human review.

### 3. Temporal Integrity
Data is split using a **Temporal Split (80/20)** rather than random shuffling to respect the time-series nature of CPS processes. This ensures the model learns the physical correlations (e.g., if Pump A turns on, Flow Meter B should rise after X seconds) without seeing the "future" during training.

---

## 🧠 Explainable AI (XAI) Deep Dive

To achieve "Trustworthy AI," the system utilizes two independent engines to explain every detection.

### 1. SHAP (SHapley Additive exPlanations)
- **The Theory**: Based on **Coalitional Game Theory**. SHAP treats each sensor reading as a "player" in a game and calculates its "payout" (contribution to the final prediction).
- **Why it's helpful**: It provides **Global Consistency**. If SHAP identifies `LIT101` as the top feature, it means that sensor is mathematically the most responsible for the anomaly based on the model's entire learned logic.
- **Local Explanation**: For a specific attack at 12:00 PM, SHAP tells the operator exactly which sensors shifted the probability from 0.0 to 0.9.

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- **The Theory**: LIME ignores the internal math of XGBoost. Instead, it takes a single attack instance, creates thousands of slightly "perturbed" versions of it, and sees how the model's prediction changes. It then builds a simple **Local Linear Surrogate Model** to explain that specific point.
- **Why it's helpful**: It provides an **External Validation**. Because LIME is model-agnostic, it acts as a "sanity check" for SHAP. 

### 3. XAI Agreement & Consensus
The system computes an **Agreement Score**:
- **Top-1 Agreement**: Do SHAP and LIME agree on the #1 most important sensor?
- **Jaccard Overlap**: How much do the Top-3 features from both methods overlap?
- **Trust Factor**: High agreement (e.g., >60%) gives the operator confidence that the explanation is a physical reality, not a mathematical artifact.

---

## 🔬 Attack Pattern Clustering (Taxonomy)

Instead of just labeling a row as "Attack", the system performs **KMeans Clustering on SHAP vectors**:
- **Why Cluster on SHAP?** Clustering on raw sensor data is often noisy. By clustering on the *Explanations*, we group attacks by their **Root Cause**.
- **Auto-Labeling**: 
    - The system compares the **Statistical Deviation** (which sensor moved most physically) with the **SHAP Importance** (which sensor the model focused on).
    - If they agree, the cluster is labeled: `[Sensor]-dominant attack`.
    - If they disagree, it's labeled: `Mixed disturbance (C/D conflict)`, indicating a complex, multi-stage attack.
- **Validation**: Uses **Silhouette Scores** to ensure clusters are distinct and meaningful.

---

## 📊 Dashboard UI Features (Fixes & Enhancements)

The `app.py` has been overbuilt for professional use:
- **📡 Key Sensor Grid**: Fixed display of the **Top-8 SHAP sensors**. Users see the exact same sensors every loop, making it easier to track process stability.
- **📈 Delta Tracking**: Every sensor metric shows a **Delta (Δ)** arrow. If `FIT101` rises, the red/green arrow shows the immediate shift from the previous second.
- **🚨 Pulse Alerts**: A high-end "Pulse-Red" animation with box-shadow effects replaces cheap blinking, providing a premium SOC (Security Operations Center) feel.
- **🗂 Session Log**: A persistent table of all detected attacks, including timestamps and the consensus between SHAP and LIME.

---

## 📁 File Manifest & Usage

### Running the System
1.  **Execute Pipeline**: `python ctmas_pipeline.py`
    - *This generates all `.json`, `.csv`, and `.png` artifacts in `/outputs`.*
2.  **Launch Dashboard**: `streamlit run app.py`

### Key Outputs in `/outputs`
- `shap_attack_clusters.png`: The PCA-reduced map of threat signatures.
- `attack_cluster_summary.csv`: The automated taxonomy of attack types.
- `xai_agreement_summary.txt`: Detailed metrics on explanation trustworthiness.
- `shap_local_explanations.csv`: The human-readable reasoning for every threat.

---

## 🏛 Project Context
**Full Title**: Explainable AI–Based Threat Modeling for Trustworthy Cyber–Physical Systems under Intelligent Adversaries

This framework demonstrates that for AI to be deployed in critical infrastructure, it must not only be **accurate** but also **interpretable**, **consistent**, and **transparent**. By combining Gradient Boosting with Dual XAI and Clustering, we provide a complete toolset for modern cyber-defense.
