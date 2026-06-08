# AI-Powered Spam Email Classifier

> **Hybrid ML + LLM routing pipeline** — a classical ensemble handles high-confidence cases instantly and for free; only ambiguous edge cases are escalated to the OpenAI API, making the system both accurate and cost-efficient at scale.

---

## Why ML + LLM Hybrid?

A pure LLM approach charges an API call for every single email — expensive and slow at volume. A pure ML approach breaks down on novel phishing patterns that weren't in the training data. This project takes the best of both:

```
                 ┌─────────────────────────────────┐
  email body ──► │   TF-IDF + NLP Preprocessing    │
                 └────────────┬────────────────────┘
                              │
                 ┌────────────▼────────────────────┐
                 │  ML Ensemble (NB · LR · SVM)    │
                 │  best model selected by CV F1   │
                 └────────────┬────────────────────┘
                              │
                   spam_prob in [0.35, 0.65]?
                    ┌─────────┴─────────┐
                   YES                  NO
                    │                   │
       ┌────────────▼──────────┐   ┌───▼─────────────────┐
       │  LLM Fallback Layer   │   │  ML Prediction       │
       │  (OpenAI gpt-4o-mini) │   │  label + confidence  │
       │  label + reason       │   └─────────────────────-┘
       └───────────────────────┘
```

| Concern | ML-only | LLM-only | **This project** |
|---|---|---|---|
| Cost per 1 M emails | ~$0 | ~$150–600 | **~$2–10** (LLM called only on ~5% ambiguous cases) |
| Latency | <5 ms | 500–2000 ms | **<5 ms** typical, 500 ms on edge cases |
| Novel phishing patterns | Weak | Strong | **Strong** (LLM arbiter on uncertain inputs) |
| Explainability | Low | High (reason field) | **High on routed cases** |

---

## Features

- **NLP preprocessing pipeline** — lowercasing, URL/HTML stripping, tokenization, stopword removal, lemmatization, TF-IDF vectorization (uni- + bi-grams)
- **Three trained classifiers** — Multinomial Naive Bayes, Logistic Regression, Linear SVM — with cross-validated F1-score selection
- **Confidence-aware routing** — ambiguous predictions (`0.35 ≤ p(spam) ≤ 0.65`) are forwarded to `gpt-4o-mini` with a structured JSON response
- **Graceful degradation** — LLM fallback is fully optional; if `OPENAI_API_KEY` is unset the ML prediction is returned without any API call
- **Pre-trained artifact** — `artifacts/best_model.pkl` is committed; zero-setup inference out of the box
- **Production CLI** — `predict.py` features structured logging, defensive error handling, and a clear recovery path if the model artifact is missing
- **Docker-ready** — multi-stage `Dockerfile` keeps the final image lean; `docker-compose.yml` for one-command deployment

---

## Project Structure

```
spam-email-classifier/
├── Dockerfile                # multi-stage production build
├── docker-compose.yml        # one-command deployment
├── requirements.txt
├── train.py                  # train + persist models
├── predict.py                # production CLI for classifying emails
├── data/
│   └── sample_emails.csv     # 20-row smoke-test set (7 spam / 7 ham / 6 ambiguous)
├── artifacts/
│   └── best_model.pkl        # pre-trained model (zero-setup inference)
├── src/
│   ├── __init__.py           # public package surface
│   ├── preprocessing.py      # text cleaning + TF-IDF
│   ├── models.py             # NB / LR / SVM trainers
│   ├── llm_fallback.py       # OpenAI API integration
│   ├── classifier.py         # orchestrates ML + LLM routing
│   └── evaluate.py           # precision / recall / F1 / ROC-AUC
└── tests/
    └── test_preprocessing.py
```

---

## Quickstart

```bash
git clone https://github.com/offninad/spam-email-classifier.git
cd spam-email-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# A pre-trained model is included — classify straight away (no training needed):
python predict.py --text "Congratulations! You won a $1000 gift card."
# Label:       SPAM
# Confidence:  0.9821
# Source:      ml

# Enable the LLM fallback for edge cases (optional):
export OPENAI_API_KEY=sk-...
python predict.py --text "Your free trial ends in 3 days. Upgrade now." --use-llm-fallback
# Label:       SPAM
# Confidence:  0.92
# Source:      llm
# Reason:      Promotional urgency with expiry language is a strong spam signal ...

# Retrain on a larger corpus (or your own CSV — must have columns: text, label):
python train.py --data data/sample_emails.csv --out artifacts/
```

---

## Production Deployment

### Docker (single container)

```bash
# Build the image
docker build --target runtime -t spam-email-classifier:latest .

# Classify inline text
docker run --rm spam-email-classifier:latest \
  --text "Your account has been suspended. Click to verify."

# With LLM fallback
docker run --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" spam-email-classifier:latest \
  --text "Your free trial expires tomorrow. Upgrade now." \
  --use-llm-fallback

# Classify from a local file
docker run --rm -v "$(pwd)/email.txt:/tmp/email.txt:ro" spam-email-classifier:latest \
  --file /tmp/email.txt
```

### Docker Compose

```bash
# Run the default smoke-test prediction
docker compose up

# Override the email text at runtime
docker compose run --rm classifier \
  --model artifacts/best_model.pkl \
  --text "Hi team, standup is at 10 AM today."

# With LLM fallback (reads OPENAI_API_KEY from your shell environment)
OPENAI_API_KEY=sk-... docker compose run --rm classifier \
  --model artifacts/best_model.pkl \
  --text "Your exclusive discount expires tonight." \
  --use-llm-fallback
```

The image uses a **multi-stage build** (builder → slim runtime). NLTK corpora are embedded at build time so inference never requires an outbound network call.

---

## Smoke-Test Dataset

`data/sample_emails.csv` ships with **20 curated rows** purpose-built for end-to-end pipeline verification:

| Rows | Category | Design intent |
|---|---|---|
| 7 | Obvious spam | Maximally clear signals (prize scams, pharma ads, crypto fraud) — should classify with high confidence via ML |
| 7 | Obvious ham | Unambiguous legitimate email (CI alerts, transactional, team comms) — should classify with high confidence via ML |
| 6 | Ambiguous edge cases | Carefully crafted to fall in the uncertain zone: phishing written like security alerts, promotional copy with transactional language, recruiter outreach — designed to trigger LLM routing |

For production-scale training, plug in a larger public corpus such as the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) or [Enron-Spam](https://www2.aueb.gr/users/ion/data/enron-spam/) dataset (same `text,label` format, drop-in compatible with `train.py`).

---

## OpenAI Fallback

The fallback is **fully optional**. Without `OPENAI_API_KEY`, the classifier silently returns the ML prediction. When enabled, ambiguous inputs are sent to `gpt-4o-mini` with a structured JSON response prompt; the model returns `label`, `confidence`, and a human-readable `reason`.

Routing thresholds are configurable at train time:

```bash
python train.py --data data/sample_emails.csv --out artifacts/ \
  --low-threshold 0.30 --high-threshold 0.70
```

---

## Evaluation

`src/evaluate.py` reports precision, recall, F1, and ROC-AUC per model during training. Selection optimises **F1** — this controls false negatives (missed spam) without sacrificing precision (legitimate mail flagged as spam), the standard production trade-off for email filters.

---

## Repository Description & Topics

> **1-line description** (use on the GitHub repo About page):
> *Hybrid spam classifier — ML ensemble routes confident cases locally; ambiguous edge cases escalate to an LLM fallback for explainable, cost-efficient decisions at scale.*

**Suggested GitHub topics:**
`machine-learning` · `nlp` · `spam-detection` · `scikit-learn` · `openai` · `llm` · `python` · `text-classification`

---

## License

MIT — see [`LICENSE`](LICENSE).
