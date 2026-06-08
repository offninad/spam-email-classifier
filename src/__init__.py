"""AI-Powered Spam Email Classifier — public package surface.

Importing from the package root (``from src import SpamClassifier``) works
from the project root directory without any ``sys.path`` manipulation.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    __version__: str = _pkg_version("spam-email-classifier")
except Exception:
    __version__ = "0.1.0"

from .classifier import Prediction, SpamClassifier
from .preprocessing import Vectorizer, clean_text

__all__ = [
    "SpamClassifier",
    "Prediction",
    "Vectorizer",
    "clean_text",
    "__version__",
]
