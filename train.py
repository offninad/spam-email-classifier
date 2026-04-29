"""Train spam classifiers on a CSV with columns: text,label.

Example:
    python train.py --data data/sample_emails.csv --out artifacts/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifier import STR_TO_LABEL, SpamClassifier
from src.evaluate import print_report
from src.models import select_best, train_all
from src.preprocessing import Vectorizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spam classifiers.")
    p.add_argument("--data", required=True, help="CSV path with columns: text,label")
    p.add_argument("--out", default="artifacts", help="Output directory.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--low-threshold", type=float, default=0.35)
    p.add_argument("--high-threshold", type=float, default=0.65)
    return p.parse_args()


def load_dataset(path: str):
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must have columns: text, label")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(STR_TO_LABEL.keys())]
    df["y"] = df["label"].map(STR_TO_LABEL)
    return df


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading {args.data}")
    df = load_dataset(args.data)
    print(f"[train] dataset size: {len(df)} | spam={int((df.y == 1).sum())} ham={int((df.y == 0).sum())}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["y"].tolist(),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["y"] if df["y"].nunique() > 1 else None,
    )

    print("[train] fitting TF-IDF vectorizer")
    vectorizer = Vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("[train] training models with cross-validated F1")
    trained = train_all(X_train, y_train)
    for tm in trained:
        print(f"  - {tm.name:22s} CV F1 = {tm.cv_f1_mean:.4f} (+/- {tm.cv_f1_std:.4f})")

    best = select_best(trained)
    print(f"[train] best model: {best.name} (CV F1 = {best.cv_f1_mean:.4f})")

    for tm in trained:
        print_report(tm.name, tm.estimator, X_test, y_test)

    classifier = SpamClassifier(
        vectorizer=vectorizer,
        estimator=best.estimator,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    out_path = out_dir / "best_model.pkl"
    classifier.save(str(out_path))
    print(f"[train] saved best model to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
