# AI-Powered Spam Email Classifier

An end-to-end spam email classification system that combines classical machine
learning with an LLM fallback. Classical models (Naive Bayes, Logistic
Regression, SVM) handle confident predictions, while ambiguous emails are
routed to the OpenAI API for a second opinion. Built as an agentic decision
pipeline.

## Features

- NLP preprocessing pipeline: lowercasing, tokenization, stopword removal,
  lemmatization, and TF-IDF vectorization
- Three trained classifiers (Multinomial Naive Bayes, Logistic Regression,
  Linear SVM) with cross-validated F1-score selection
- Confidence-aware routing: when the best classifier's predicted probability
  falls below a threshold, the email is forwarded to the OpenAI API
- CLI for training (`train.py`) and prediction (`predict.py`)
- Pluggable design — preprocessing, models, and the LLM fallback are independent
  modules

## Project Structure

```
spam-email-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── train.py                  # train + persist models
├── predict.py                # CLI for classifying new emails
├── data/
│   └── sample_emails.csv     # small synthetic dataset for smoke-testing
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # text cleaning + TF-IDF
│   ├── models.py             # NB / LR / SVM trainers
│   ├── llm_fallback.py       # OpenAI API integration
│   ├── classifier.py         # orchestrates ML + LLM routing
│   └── evaluate.py           # precision / recall / F1 / ROC-AUC
└── tests/
    └── test_preprocessing.py
```

## Quickstart

```bash
git clone https://github.com/offninad/spam-email-classifier.git
cd spam-email-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# a pre-trained model is included — classify straight away:
python predict.py --model artifacts/best_model.pkl --text "Congratulations! You won a $1000 gift card."

# or retrain on the bundled 1,000-example dataset (or your own CSV: text,label):
python train.py --data data/sample_emails.csv --out artifacts/
```

## Dataset

`data/sample_emails.csv` ships with ~1,000 labelled examples (balanced spam/ham)
covering common patterns: phishing, prize scams, drug ads, financial fraud, and
realistic ham (work email, calendar reminders, pull requests, etc.). It is large
enough to train a model that generalises reasonably well out of the box.

For production-scale accuracy, swap in a larger public corpus such as the
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
or the [Enron-Spam](https://www2.aueb.gr/users/ion/data/enron-spam/) dataset
(both use the same two-column `text,label` format accepted by `train.py`).

## OpenAI Fallback

The fallback is optional. If `OPENAI_API_KEY` is not set in the environment,
the classifier returns the ML model's prediction without calling the API.
To enable it:

```bash
export OPENAI_API_KEY=sk-...
python predict.py --text "..." --use-llm-fallback
```

## Evaluation

`src/evaluate.py` reports precision, recall, F1, and ROC-AUC for each model
during training. Model selection optimizes F1 to control false negatives
(missed spam) without sacrificing precision (legit mail flagged as spam).

## License

MIT — see `LICENSE`.
