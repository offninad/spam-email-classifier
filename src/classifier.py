"""High-level classifier that orchestrates ML + optional LLM fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib

from . import llm_fallback
from .preprocessing import Vectorizer


# Mapping between integer labels stored by sklearn and string labels.
LABEL_TO_STR = {0: "ham", 1: "spam"}
STR_TO_LABEL = {"ham": 0, "spam": 1}


@dataclass
class Prediction:
    label: str          # 'spam' or 'ham'
    probability: float  # confidence in the predicted class
    source: str         # 'ml' or 'llm'
    detail: Optional[str] = None


@dataclass
class SpamClassifier:
    """Bundle vectorizer + estimator + thresholds.

    The two thresholds define the 'ambiguous zone'. If the ML model's
    probability for 'spam' falls inside [low, high], the LLM fallback is
    consulted (when available).
    """

    vectorizer: Vectorizer
    estimator: object
    low_threshold: float = 0.35
    high_threshold: float = 0.65
    use_llm_fallback: bool = False

    # ---- predict ---------------------------------------------------------

    def predict(self, text: str) -> Prediction:
        X = self.vectorizer.transform([text])
        # Try predict_proba; fall back to decision_function for SVM.
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)[0]
            spam_p = float(proba[STR_TO_LABEL["spam"]])
        else:
            score = float(self.estimator.decision_function(X)[0])
            # crude logistic squash
            spam_p = 1.0 / (1.0 + pow(2.71828, -score))

        ambiguous = self.low_threshold <= spam_p <= self.high_threshold
        if ambiguous and self.use_llm_fallback and llm_fallback.is_available():
            verdict = llm_fallback.classify(text)
            if verdict is not None:
                return Prediction(
                    label=verdict.label,
                    probability=verdict.confidence,
                    source="llm",
                    detail=verdict.reason,
                )

        label = "spam" if spam_p >= 0.5 else "ham"
        confidence = spam_p if label == "spam" else 1.0 - spam_p
        return Prediction(label=label, probability=confidence, source="ml")

    # ---- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "estimator": self.estimator,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, use_llm_fallback: bool = False) -> "SpamClassifier":
        bundle = joblib.load(path)
        return cls(
            vectorizer=bundle["vectorizer"],
            estimator=bundle["estimator"],
            low_threshold=bundle.get("low_threshold", 0.35),
            high_threshold=bundle.get("high_threshold", 0.65),
            use_llm_fallback=use_llm_fallback,
        )
