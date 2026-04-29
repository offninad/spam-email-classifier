"""Text preprocessing and TF-IDF feature extraction.

Pipeline steps:
    1. Lowercase
    2. Strip URLs, HTML tags, numbers, punctuation
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    6. TF-IDF vectorize (uni- + bi-grams)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable, List

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NUM_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# Compact built-in fallback. Used when the NLTK stopword corpus isn't
# downloaded (e.g. on first run with no internet, or in CI sandboxes).
_FALLBACK_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can could did do does doing don down during
each few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours ourselves out over own re s same she should so some
such t than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves
""".split())


def _ensure_nltk_resources() -> None:
    """Best-effort download. Silent if it fails — we have fallbacks."""
    for pkg, path in (
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("wordnet", "corpora/wordnet"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk_resources()


def _load_stopwords() -> frozenset:
    try:
        from nltk.corpus import stopwords as _sw
        return frozenset(_sw.words("english"))
    except Exception:
        return _FALLBACK_STOPWORDS


def _make_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        lemmer = WordNetLemmatizer()
        # smoke test — will raise LookupError if wordnet is missing
        lemmer.lemmatize("running")
        return lemmer.lemmatize
    except Exception:
        # Identity fallback — preprocessing degrades gracefully.
        return lambda tok: tok


def _make_tokenizer():
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("hello world")
        return word_tokenize
    except Exception:
        return lambda text: text.split()


_STOPWORDS = _load_stopwords()
_LEMMATIZE = _make_lemmatizer()
_TOKENIZE = _make_tokenizer()


def clean_text(text: str) -> str:
    """Apply normalization to a single email body."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = _TOKENIZE(text)
    tokens = [
        _LEMMATIZE(tok)
        for tok in tokens
        if tok and tok not in _STOPWORDS and len(tok) > 1
    ]
    return " ".join(tokens)


def clean_corpus(texts: Iterable[str]) -> List[str]:
    return [clean_text(t) for t in texts]


@dataclass
class Vectorizer:
    """Thin wrapper around TfidfVectorizer with sensible defaults."""

    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    min_df: int = 1

    def __post_init__(self) -> None:
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: Iterable[str]):
        cleaned = clean_corpus(texts)
        return self._vec.fit_transform(cleaned)

    def transform(self, texts: Iterable[str]):
        cleaned = clean_corpus(texts)
        return self._vec.transform(cleaned)

    @property
    def sklearn_vectorizer(self) -> TfidfVectorizer:
        return self._vec
