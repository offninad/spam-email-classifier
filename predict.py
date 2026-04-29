"""Classify a single email from the command line.

Examples:
    python predict.py --model artifacts/best_model.pkl --text "Win a free iPhone now!"
    python predict.py --model artifacts/best_model.pkl --file my_email.txt --use-llm-fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.classifier import SpamClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify an email body as spam or ham.")
    p.add_argument("--model", required=True, help="Path to a trained .pkl bundle.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Email body to classify (inline).")
    g.add_argument("--file", help="Path to a file containing the email body.")
    p.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Forward ambiguous cases to the OpenAI API (requires OPENAI_API_KEY).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    body = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")

    classifier = SpamClassifier.load(args.model, use_llm_fallback=args.use_llm_fallback)
    result = classifier.predict(body)

    print(f"Label:       {result.label.upper()}")
    print(f"Confidence:  {result.probability:.4f}")
    print(f"Source:      {result.source}")
    if result.detail:
        print(f"Reason:      {result.detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
