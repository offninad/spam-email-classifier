"""OpenAI fallback for ambiguous classifications.

Used when the best ML model's predicted probability is between
[low_threshold, high_threshold]. The LLM acts as an arbiter, returning
'spam' or 'ham' along with a short rationale.

Gracefully degrades when OPENAI_API_KEY is not set: returns None so the
caller falls back to the ML prediction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # openai is optional for offline use
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = """You are an email security classifier. Given the body of an
email, decide if it is "spam" or "ham" (legitimate). Respond ONLY with a JSON
object of the form: {"label": "spam"|"ham", "confidence": 0.0-1.0, "reason": "..."}.
Be decisive — pick the more likely label even when uncertain."""


@dataclass
class LLMVerdict:
    label: str
    confidence: float
    reason: str
    raw_response: str


def is_available() -> bool:
    return OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))


def classify(text: str, model: str = "gpt-4o-mini") -> Optional[LLMVerdict]:
    """Call the OpenAI Chat API to classify an email body.

    Returns ``None`` if the API key is missing or the call fails.
    """
    if not is_available():
        return None

    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:4000]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
    except Exception as exc:  # network / quota / model-not-found
        print(f"[llm_fallback] OpenAI call failed: {exc}")
        return None

    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    label = str(data.get("label", "")).lower().strip()
    if label not in {"spam", "ham"}:
        return None
    return LLMVerdict(
        label=label,
        confidence=float(data.get("confidence", 0.5)),
        reason=str(data.get("reason", "")),
        raw_response=raw,
    )
