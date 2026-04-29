"""Three classical classifiers wired up with consistent training/eval interfaces.

Models:
    - Multinomial Naive Bayes
    - Logistic Regression
    - Linear SVM (calibrated for probability output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def get_model_zoo() -> Dict[str, object]:
    """Return a fresh dict of {name: untrained estimator}."""
    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="liblinear"
        ),
        # CalibratedClassifierCV gives LinearSVC predict_proba support.
        "linear_svm": CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
    }


@dataclass
class TrainedModel:
    name: str
    estimator: object
    cv_f1_mean: float
    cv_f1_std: float
    extras: Dict[str, float] = field(default_factory=dict)


def train_all(X, y, cv_splits: int = 5) -> List[TrainedModel]:
    """Train every model in the zoo; return them with cross-validated F1 stats."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    trained: List[TrainedModel] = []

    for name, estimator in get_model_zoo().items():
        try:
            scores = cross_val_score(estimator, X, y, cv=cv, scoring="f1", n_jobs=-1)
        except Exception:
            # Tiny datasets may not support cv_splits folds — fall back gracefully.
            scores = np.array([0.0])
        estimator.fit(X, y)
        trained.append(
            TrainedModel(
                name=name,
                estimator=estimator,
                cv_f1_mean=float(scores.mean()),
                cv_f1_std=float(scores.std()),
            )
        )
    return trained


def select_best(trained: List[TrainedModel]) -> TrainedModel:
    """Pick the model with the best mean CV F1."""
    return max(trained, key=lambda m: m.cv_f1_mean)
