"""
model/train.py
Trains the Amazon Review Analyzer classifier and saves the pipeline.

Usage:
    python model/train.py
    python model/train.py --data data/reviews_sample.csv --output model/artifacts/pipeline.pkl
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from features import build_feature_matrix, label_from_rating

ARTIFACTS_DIR = "model/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return a cleaned DataFrame."""
    print(f"[train] Loading data from {path} ...")
    df = pd.read_csv(path)

    required = {"Text", "Score"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    df = df.dropna(subset=["Text", "Score"])
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score"])
    df["Score"] = df["Score"].clip(1, 5)

    # Label encoding
    df["label"] = df["Score"].apply(label_from_rating)
    print(f"[train] Loaded {len(df):,} reviews.")
    print(df["label"].value_counts().to_string())
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, model_type: str = "rf"):
    print(f"\n[train] Building features ...")
    X, feature_names = build_feature_matrix(df["Text"], df["Score"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        print("[train] Training Logistic Regression (baseline) ...")
    else:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        print("[train] Training Random Forest ...")

    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_test, y_pred))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Caution", "Don't Buy"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Buy", "Caution", "Don't Buy"],
                yticklabels=["Buy", "Caution", "Don't Buy"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    print(f"[train] Confusion matrix saved → {cm_path}")

    return clf, feature_names, macro_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/reviews_sample.csv", help="Path to reviews CSV")
    parser.add_argument("--output", default=f"{ARTIFACTS_DIR}/pipeline.pkl", help="Output model path")
    parser.add_argument("--model", choices=["rf", "lr"], default="rf", help="Model type")
    args = parser.parse_args()

    df = load_data(args.data)
    clf, feature_names, f1 = train(df, model_type=args.model)

    # Save model bundle
    bundle = {
        "classifier": clf,
        "feature_names": feature_names,
        "macro_f1": f1,
        "model_type": args.model,
    }
    joblib.dump(bundle, args.output)
    print(f"\n[train] Model saved → {args.output}  (Macro F1={f1:.4f})")


if __name__ == "__main__":
    main()
