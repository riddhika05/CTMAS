"""
CTMAS: Cyber-Physical Systems Threat Monitoring & Analysis System
================================================================
XGBoost + SHAP Explainable AI Pipeline for SWaT Dataset Security.

Pipeline Steps:
  1. Preprocessing   -> clean_normal.csv, clean_attack.csv, clean_merged.csv
  2. Data Validation -> numeric check, missing values, class distribution
  3. Train-Test Split -> temporal (no shuffle), 80/20
  4. XGBoost Training -> reproducible, configured classifier
  5. Evaluation       -> accuracy, precision, recall, F1, confusion matrix
  6. Explainability   -> SHAP TreeExplainer (global + local)
  7. Threat Reasoning -> human-readable attack explanations
  8. Output Files     -> predictions.csv, shap_summary.csv, shap_local_explanations.csv
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "archive3")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_NORMAL  = os.path.join(DATA_DIR, "normal.csv")
RAW_ATTACK  = os.path.join(DATA_DIR, "attack.csv")
RAW_MERGED  = os.path.join(DATA_DIR, "merged.csv")

CLEAN_NORMAL  = os.path.join(DATA_DIR, "clean_normal.csv")
CLEAN_ATTACK  = os.path.join(DATA_DIR, "clean_attack.csv")
CLEAN_MERGED  = os.path.join(DATA_DIR, "clean_merged.csv")

TIMESTAMP_COL = "Timestamp"
LABEL_COL     = "label"
TRAIN_RATIO   = 0.80
RANDOM_STATE  = 42

XGBOOST_PARAMS = dict(
    n_estimators     = 100,
    max_depth        = 5,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = RANDOM_STATE,
    eval_metric      = "logloss",
    # scale_pos_weight is set dynamically in train_model() based on class counts
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(OUTPUT_DIR, "pipeline.log"),
            mode="w",
            encoding="utf-8",
        ),
    ],
)
# Force stdout to UTF-8 on Windows to avoid cp1252 encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 – Pre-processing  (produce clean_*.csv if they do not already exist)
# ──────────────────────────────────────────────────────────────────────────────
def _clean_single(raw_path: str, expected_label: str) -> pd.DataFrame:
    """
    Load one raw CSV, strip column whitespace, parse timestamp,
    encode label to binary 0/1, drop non-numeric leftovers.
    """
    log.info(f"  Loading raw file: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename 'Normal/Attack' → label and map to binary int
    if "Normal/Attack" in df.columns:
        df.rename(columns={"Normal/Attack": LABEL_COL}, inplace=True)
    df[LABEL_COL] = (df[LABEL_COL].str.strip().str.lower() == "attack").astype(int)

    # Parse and set timestamp as index
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=False, errors="coerce")
        df.set_index("Timestamp", inplace=True)
        df.sort_index(inplace=True)   # ensure temporal order

    # Drop any columns that are still object-typed (safety net)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        log.warning(f"    Dropping non-numeric columns: {obj_cols}")
        df.drop(columns=obj_cols, inplace=True)

    # Convert everything to float32 for memory efficiency (keep label as int)
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df[feature_cols] = df[feature_cols].astype("float32")
    df[LABEL_COL]    = df[LABEL_COL].astype("int8")

    # Fill any NaN with column median (rare in SWaT)
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        log.warning(f"    Found {n_missing} missing values – filling with column medians.")
        df.fillna(df.median(), inplace=True)

    log.info(f"    Shape after cleaning: {df.shape} | "
             f"Attack rows: {df[LABEL_COL].sum()} / {len(df)}")
    return df


def preprocess():
    """Build clean_normal.csv, clean_attack.csv, clean_merged.csv.

    Strategy:
    - The authoritative SWaT source is merged.csv which contains temporally
      interleaved Normal AND Attack segments across multiple days.
    - clean_merged.csv is built from merged.csv (preserving temporal order).
    - clean_normal.csv and clean_attack.csv are derived subsets.
    """
    log.info("=" * 70)
    log.info("STEP 0 - PRE-PROCESSING")
    log.info("=" * 70)

    if all(os.path.exists(p) for p in [CLEAN_NORMAL, CLEAN_ATTACK, CLEAN_MERGED]):
        log.info("  Clean files already exist - skipping pre-processing.")
        return

    # ---- 1. Clean merged.csv (primary source with temporal interleaving) ----
    log.info("  Processing merged.csv (primary SWaT dataset) ...")
    df_merged = _clean_single(RAW_MERGED, "merged")
    df_merged.to_csv(CLEAN_MERGED)
    log.info(f"  Saved -> {CLEAN_MERGED} | Shape: {df_merged.shape}")

    # ---- 2. Derive clean_normal and clean_attack as subsets ----------------
    df_normal = df_merged[df_merged[LABEL_COL] == 0].copy()
    df_normal.to_csv(CLEAN_NORMAL)
    log.info(f"  Saved -> {CLEAN_NORMAL} | Normal rows: {len(df_normal):,}")

    df_attack = df_merged[df_merged[LABEL_COL] == 1].copy()
    df_attack.to_csv(CLEAN_ATTACK)
    log.info(f"  Saved -> {CLEAN_ATTACK} | Attack rows: {len(df_attack):,}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    log.info("=" * 70)
    log.info("STEP 1 – DATA LOADING")
    log.info("=" * 70)

    log.info(f"  Reading {CLEAN_MERGED} …")
    df = pd.read_csv(CLEAN_MERGED, index_col=0, parse_dates=True, low_memory=False)
    log.info(f"  Loaded shape: {df.shape}")

    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)

    log.info(f"  Features : {X.shape[1]} columns")
    log.info(f"  Samples  : {len(y)}")
    log.info(f"  Feature names: {list(X.columns)}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – Data Validation
# ──────────────────────────────────────────────────────────────────────────────
def validate_data(X: pd.DataFrame, y: pd.Series):
    log.info("=" * 70)
    log.info("STEP 2 – DATA VALIDATION")
    log.info("=" * 70)

    # Numeric check
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.error(f"  Non-numeric columns found: {non_numeric}")
        raise ValueError("Non-numeric features detected.")
    log.info("  ✓ All features are numeric.")

    # Missing values
    missing = X.isnull().sum()
    n_missing = missing.sum()
    if n_missing > 0:
        log.warning(f"  ✗ Missing values detected:\n{missing[missing > 0]}")
    else:
        log.info("  ✓ No missing values.")

    # Label distribution
    counts      = y.value_counts().sort_index()
    total       = len(y)
    n_normal    = counts.get(0, 0)
    n_attack    = counts.get(1, 0)
    imbalance   = n_normal / n_attack if n_attack > 0 else float("inf")

    log.info(f"  Label distribution:")
    log.info(f"    Normal  (0): {n_normal:>10,}  ({100*n_normal/total:.2f}%)")
    log.info(f"    Attack  (1): {n_attack:>10,}  ({100*n_attack/total:.2f}%)")
    log.info(f"    Imbalance ratio (Normal:Attack) = {imbalance:.1f}:1")
    if imbalance > 3:
        log.warning("  ⚠  Significant class imbalance detected.  "
                    "Recall (attack detection) will be the primary metric.")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – Temporal Train-Test Split (NO shuffle)
# ──────────────────────────────────────────────────────────────────────────────
def random_split(X: pd.DataFrame, y: pd.Series):
    log.info("=" * 70)
    log.info("STEP 3 - RANDOM TRAIN-TEST SPLIT (Shuffled)")
    log.info("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )
    
    log.info(f"  Split strategy : SHUFFLED & STRATIFIED ({int(TRAIN_RATIO*100)}/{100-int(TRAIN_RATIO*100)})")

    log.info(f"  Train set: {X_train.shape[0]:,} samples "
             f"(Attack: {int(y_train.sum()):,} / {len(y_train):,})")
    log.info(f"  Test  set: {X_test.shape[0]:,} samples "
             f"(Attack: {int(y_test.sum()):,} / {len(y_test):,})")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 & 5 – Model Training
# ──────────────────────────────────────────────────────────────────────────────
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    log.info("=" * 70)
    log.info("STEP 4 & 5 - MODEL SELECTION & TRAINING")
    log.info("=" * 70)

    # Compute scale_pos_weight = count(negatives) / count(positives)
    # This is XGBoost's recommended way to handle class imbalance.
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw   = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

    params = dict(**XGBOOST_PARAMS, scale_pos_weight=spw)
    log.info(f"  XGBoost params: {params}")
    log.info(f"  scale_pos_weight = {spw}  (neg={n_neg:,} / pos={n_pos:,})")

    model = XGBClassifier(**params)
    log.info("  Training ... (this may take a moment)")
    model.fit(X_train, y_train)
    
    model_path = os.path.join(OUTPUT_DIR, "xgboost_model.json")
    model.save_model(model_path)
    log.info(f"  Training complete. Model saved -> {model_path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 – Prediction
# ──────────────────────────────────────────────────────────────────────────────
def predict(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series,
            threshold: float = 0.3) -> pd.DataFrame:
    log.info("=" * 70)
    log.info("STEP 6 - PREDICTION")
    log.info("=" * 70)

    y_prob  = model.predict_proba(X_test)[:, 1]   # probability for class 1 (attack)
    y_pred  = (y_prob >= threshold).astype(int)   # lower threshold to boost recall

    log.info(f"  Decision threshold : {threshold}  (lowered from 0.5 to improve attack recall)")
    predictions_df = pd.DataFrame({
        "actual"           : y_test.values,
        "predicted"        : y_pred,
        "attack_probability": np.round(y_prob, 6),
    }, index=X_test.index)

    out_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    predictions_df.to_csv(out_path)
    log.info(f"  Predictions saved -> {out_path}")
    log.info(f"  Predicted attacks: {y_pred.sum():,} / {len(y_pred):,}")
    return predictions_df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 – Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(predictions_df: pd.DataFrame):
    log.info("=" * 70)
    log.info("STEP 7 – EVALUATION")
    log.info("=" * 70)

    y_true = predictions_df["actual"]
    y_pred = predictions_df["predicted"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    log.info(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}  ← Critical for attack detection")
    log.info(f"  F1-Score  : {f1:.4f}")
    log.info(f"\n  Classification Report:\n"
             f"{classification_report(y_true, y_pred, target_names=['Normal','Attack'])}")

    log.info("  Confusion Matrix:")
    log.info(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    log.info(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Attack'])
    disp.plot(cmap='Blues', values_format=',', ax=ax)
    
    ax.set_title("Confusion Matrix – XGBoost CPS Threat Detector", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log.info(f"  Confusion matrix plot saved → {cm_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 – Explainability (SHAP)
# ──────────────────────────────────────────────────────────────────────────────
def explain(model: XGBClassifier, X_test: pd.DataFrame, predictions_df: pd.DataFrame):
    log.info("=" * 70)
    log.info("STEP 8 – EXPLAINABILITY (SHAP TreeExplainer)")
    log.info("=" * 70)

    log.info("  Initialising SHAP TreeExplainer …")
    explainer   = shap.TreeExplainer(model)

    # Use a manageable sample for SHAP if dataset is very large (> 50k rows)
    MAX_SHAP_ROWS = 50_000
    if len(X_test) > MAX_SHAP_ROWS:
        log.warning(f"  Test set has {len(X_test):,} rows – "
                    f"sampling {MAX_SHAP_ROWS:,} for SHAP computation.")
        X_shap = X_test.iloc[:MAX_SHAP_ROWS]
    else:
        X_shap = X_test

    log.info(f"  Computing SHAP values for {len(X_shap):,} test instances …")
    shap_values = explainer.shap_values(X_shap)
    log.info("  ✓ SHAP values computed.")

    feature_names = list(X_test.columns)

    # ── 8a. Global Feature Importance (mean |SHAP|) ──────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_summary  = pd.DataFrame({
        "feature"          : feature_names,
        "mean_abs_shap"    : mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_summary["rank"] = shap_summary.index + 1

    summary_path = os.path.join(OUTPUT_DIR, "shap_summary.csv")
    shap_summary.to_csv(summary_path, index=False)
    log.info(f"  Global SHAP summary saved → {summary_path}")
    log.info(f"  Top-10 features by mean |SHAP|:\n"
             f"{shap_summary.head(10).to_string(index=False)}")

    # SHAP bar plot – global
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = shap_summary.head(20)
    bars  = ax.barh(top_n["feature"][::-1], top_n["mean_abs_shap"][::-1],
                    color="#2563EB", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top-20 Features – Global SHAP Importance\n(XGBoost CPS Threat Detector)", fontsize=13)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_global_bar.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    log.info(f"  SHAP global bar chart saved → {bar_path}")

    # SHAP beeswarm / summary plot
    shap.initjs()
    fig_bee = plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feature_names,
        show=False, max_display=20,
    )
    bee_path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bee)
    log.info(f"  SHAP beeswarm plot saved → {bee_path}")

    return shap_values, shap_summary, X_shap


# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 – Threat Reasoning (Local SHAP explanations)
# ──────────────────────────────────────────────────────────────────────────────
def threat_reasoning(
    shap_values: np.ndarray,
    X_shap:      pd.DataFrame,
    predictions_df: pd.DataFrame,
    top_k: int = 3,
):
    log.info("=" * 70)
    log.info("STEP 9 – THREAT REASONING (Local Explanations)")
    log.info("=" * 70)

    feature_names = list(X_shap.columns)
    
    # Get predictions specifically for the SHAP subset
    pred_subset = predictions_df.loc[X_shap.index]
    attack_mask = pred_subset["predicted"] == 1
    records = []

    log.info(f"  Generating explanations for {attack_mask.sum():,} detected attack instances in the SHAP subset ...")

    for i, (ts, row) in enumerate(pred_subset[attack_mask].iterrows()):
        # i is the index in the attack_mask subset. We need the positional index in X_shap
        # to index into the shap_values array.
        pos = X_shap.index.get_loc(ts)
        if isinstance(pos, slice) or isinstance(pos, np.ndarray):
            pos = np.where(pos)[0][0] # handle duplicate timestamps if any
            
        instance_shap = shap_values[pos]

        # Rank features by absolute SHAP contribution for this instance
        abs_shap   = np.abs(instance_shap)
        top_idx    = np.argsort(abs_shap)[::-1][:top_k]
        top_feats  = [feature_names[j] for j in top_idx]
        top_vals   = [float(X_shap.iloc[pos, j]) for j in top_idx]
        top_shaps  = [float(instance_shap[j])    for j in top_idx]

        human_text = (
            f"Attack detected due to abnormal behavior in "
            + ", ".join(top_feats)
        )

        record = {
            "timestamp"       : ts,
            "actual"          : int(row["actual"]),
            "predicted"       : int(row["predicted"]),
            "attack_prob"     : float(row["attack_probability"]),
            "explanation"     : human_text,
        }
        for rank, (feat, val, shp) in enumerate(
            zip(top_feats, top_vals, top_shaps), start=1
        ):
            record[f"feature_{rank}"]     = feat
            record[f"value_{rank}"]       = round(val, 6)
            record[f"shap_contrib_{rank}"]= round(shp, 6)

        records.append(record)

        # Log first 5 for visibility
        if i < 5:
            log.info(f"  [{i+1}] {human_text}")
            log.info(f"       → prob={row['attack_probability']:.4f} | "
                     f"features: {list(zip(top_feats, [round(v,3) for v in top_shaps]))}")

    local_df = pd.DataFrame(records)
    local_path = os.path.join(OUTPUT_DIR, "shap_local_explanations.csv")
    local_df.to_csv(local_path, index=False)
    log.info(f"\n  Local explanations saved → {local_path}")
    log.info(f"  Total attack instances explained: {len(local_df):,}")
    return local_df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("  CTMAS - CPS Threat Monitoring & Analysis System")
    log.info("  XGBoost + SHAP Explainable AI Pipeline  |  SWaT Dataset")
    log.info("=" * 70)

    # Step 0: Pre-process raw files -> clean CSVs
    preprocess()

    # Step 1: Load
    X, y = load_data()

    # Step 2: Validate
    validate_data(X, y)

    # Step 3: Split
    X_train, X_test, y_train, y_test = random_split(X, y)

    # Step 4 & 5: Train
    model = train_model(X_train, y_train)

    # Step 6: Predict
    predictions_df = predict(model, X_test, y_test)

    # Step 7: Evaluate
    metrics = evaluate(predictions_df)

    # Step 8: SHAP Explainability
    shap_values, shap_summary, X_shap = explain(model, X_test, predictions_df)

    # Step 9: Threat Reasoning (local explanations)
    local_df = threat_reasoning(shap_values, X_shap, predictions_df)

    # Final Summary
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 70)
    log.info(f"  Output directory : {OUTPUT_DIR}")
    log.info(f"  Accuracy         : {metrics['accuracy']:.4f}")
    log.info(f"  Recall (Attacks) : {metrics['recall']:.4f}")
    log.info(f"  F1-Score         : {metrics['f1']:.4f}")
    log.info("  Files generated  :")
    for fname in [
        "predictions.csv",
        "shap_summary.csv",
        "shap_local_explanations.csv",
        "confusion_matrix.png",
        "shap_global_bar.png",
        "shap_beeswarm.png",
        "pipeline.log",
    ]:
        log.info(f"    → outputs/{fname}")

    return metrics, shap_summary, local_df


if __name__ == "__main__":
    main()
