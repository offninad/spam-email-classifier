# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip --quiet && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/offninad/spam-email-classifier"
LABEL org.opencontainers.image.description="AI-Powered Spam Email Classifier — ML ensemble + LLM fallback"

WORKDIR /app

# Pull installed packages from the builder — no build tools in the final image.
COPY --from=builder /install /usr/local

# Copy application source and the pre-trained model artifact.
COPY src/           ./src/
COPY predict.py     ./
COPY train.py       ./
COPY data/          ./data/
COPY artifacts/     ./artifacts/

# Pre-download NLTK corpora at build time so the container never needs internet
# at inference time. If a newer NLTK ships punkt_tab, download that too.
RUN python - <<'EOF'
import nltk
for pkg in ("stopwords", "punkt", "wordnet", "punkt_tab", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
EOF

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Validate the pre-trained artifact is loadable at build time.
RUN python -c "from src.classifier import SpamClassifier; SpamClassifier.load('artifacts/best_model.pkl'); print('[ok] model artifact verified')"

# Health-check: confirm the model can be loaded in a running container.
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from src.classifier import SpamClassifier; SpamClassifier.load('artifacts/best_model.pkl')" || exit 1

ENTRYPOINT ["python", "predict.py"]
CMD ["--help"]
