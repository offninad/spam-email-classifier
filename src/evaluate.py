"""Evaluation utilities — printable classification reports per model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def evaluate(estimator, X_test, y_test) -> Metrics:
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)
    else:
        y_score = y_pred

    try:
        auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        auc = float("nan")

    return Metrics(
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=auc,
    )


def print_report(name: str, estimator, X_test, y_test) -> None:
    print(f"\n=== {name} ===")
    print(classification_report(y_test, estimator.predict(X_test), zero_division=0))
    metrics = evaluate(estimator, X_test, y_test)
    print(f"ROC-AUC: {metrics.roc_auc:.4f}")
