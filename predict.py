"""Classify a single email from the command line.

Examples:
    python predict.py --text "Win a free iPhone now!"
    python predict.py --model artifacts/best_model.pkl --file email.txt --use-llm-fallback
    python predict.py --text "Hello team, standup at 10 AM today."
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.classifier import SpamClassifier

_DEFAULT_MODEL = Path("artifacts/best_model.pkl")

logging.basicConfig(
    format="%(levelname)s: %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify an email body as spam or ham.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python predict.py --text "Win a free iPhone now!"
  python predict.py --file email.txt --use-llm-fallback
  python predict.py --model path/to/model.pkl --text "Hello team"
""",
    )
    p.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help=f"Path to a trained .pkl bundle (default: {_DEFAULT_MODEL}).",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Email body to classify (inline string).")
    g.add_argument("--file", help="Path to a plain-text file containing the email body.")
    p.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Route ambiguous cases to the OpenAI API (requires OPENAI_API_KEY).",
    )
    return p.parse_args()


def _read_body(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text

    file_path = Path(args.file)
    if not file_path.exists():
        log.error("Input file not found: '%s'", file_path)
        sys.exit(1)
    try:
        return file_path.read_text(encoding="utf-8")
    except OSError as exc:
        log.error("Could not read '%s': %s", file_path, exc)
        sys.exit(1)


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(
            "Model artifact not found: '%s'\n"
            "       Run the following command to generate it:\n\n"
            "           python train.py --data data/sample_emails.csv --out artifacts/\n\n"
            "       Then retry.",
            model_path,
        )
        return 1

    try:
        classifier = SpamClassifier.load(
            str(model_path), use_llm_fallback=args.use_llm_fallback
        )
    except Exception as exc:
        log.error("Failed to load model from '%s': %s", model_path, exc)
        return 1

    body = _read_body(args)
    if not body.strip():
        log.warning("Input text is empty — predicting on blank input.")

    try:
        result = classifier.predict(body)
    except Exception as exc:
        log.error("Prediction failed: %s", exc)
        return 1

    print(f"Label:       {result.label.upper()}")
    print(f"Confidence:  {result.probability:.4f}")
    print(f"Source:      {result.source}")
    if result.detail:
        print(f"Reason:      {result.detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
