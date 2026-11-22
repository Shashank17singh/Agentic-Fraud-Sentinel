"""
Preliminary fake-news screening: TF-IDF features and logistic regression trained
on the ISOT Fake and Real News dataset.

The label is a first screen, not a verdict. Scores between the two cut-offs are
reported as "uncertain", and very short texts are not scored at all.

Train with:  python -m src.news.classifier data/raw/isot
"""

import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "notebooks",
    "data",
    "models",
    "news_classifier.pkl",
)

MIN_WORDS = 30
# shortcut: fixed band around 0.5, not calibrated; tune on a labelled set from the target domain.
FAKE_ABOVE = 0.65
REAL_BELOW = 0.35

# Every real ISOT article comes from Reuters and opens with a "CITY (Reuters) -"
# byline, and many fake ones carry "Featured image via" credits and links. Left in,
# the model learns the publisher instead of the writing, so these cues are removed.
_PUBLISHER_CUES = re.compile(
    r"^.{0,120}?\(reuters\)\s*-\s*|reuters|featured image via.*$|https?://\S+|pic\.twitter\.com/\S+|@\w+",
    flags=re.IGNORECASE | re.MULTILINE,
)


def clean(text: str) -> str:
    """Strip publisher cues so the model has to score the text itself."""
    return re.sub(r"\s+", " ", _PUBLISHER_CUES.sub(" ", text)).strip()


def load_isot(folder: str) -> pd.DataFrame:
    """Load Fake.csv and True.csv, clean them, and drop empty and duplicate texts."""
    fake = pd.read_csv(os.path.join(folder, "Fake.csv")).assign(label=1)
    real = pd.read_csv(os.path.join(folder, "True.csv")).assign(label=0)
    df = pd.concat([fake, real], ignore_index=True)
    df["clean"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean)
    df = df[df["text"].fillna("").str.split().str.len() >= MIN_WORDS]
    return df.drop_duplicates("clean").reset_index(drop=True)


def train(folder: str) -> dict:
    """Train the pipeline, score it on a held-out 20%, and save it with its metrics."""
    df = load_isot(folder)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    pipeline = make_pipeline(
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=50_000,
            sublinear_tf=True,
        ),
        LogisticRegression(max_iter=1000),
    )
    pipeline.fit(X_train, y_train)

    prob = pipeline.predict_proba(X_test)[:, 1]
    sure = (prob >= FAKE_ABOVE) | (prob <= REAL_BELOW)
    metrics = {
        "dataset": "ISOT Fake and Real News",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": round(accuracy_score(y_test, prob >= 0.5), 4),
        "f1": round(f1_score(y_test, prob >= 0.5), 4),
        "auc_roc": round(roc_auc_score(y_test, prob), 4),
        "uncertain_share": round(float(1 - sure.mean()), 4),
        "accuracy_when_sure": round(accuracy_score(y_test[sure], prob[sure] >= 0.5), 4),
    }
    bundle = {"pipeline": pipeline, "metrics": metrics}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    return metrics


def screen(bundle: dict, text: str) -> dict:
    """
    Give a preliminary label for a news text.

    Returns the label ("likely fake", "likely real" or "uncertain"), the fake
    probability (None when the text is too short to score), and a short note.
    """
    words = len(text.split())
    if words < MIN_WORDS:
        return {
            "label": "uncertain",
            "fake_probability": None,
            "note": f"Only {words} words. Paste at least {MIN_WORDS} words of the article.",
        }
    p = float(bundle["pipeline"].predict_proba([clean(text)])[0, 1])
    if p >= FAKE_ABOVE:
        label = "likely fake"
    elif p <= REAL_BELOW:
        label = "likely real"
    else:
        label = "uncertain"
    return {
        "label": label,
        "fake_probability": round(p, 4),
        "note": (
            "Preliminary screen from writing style only. It does not check facts; "
            "verify with a fact-checker before sharing."
        ),
    }


if __name__ == "__main__":
    print(train(sys.argv[1] if len(sys.argv) > 1 else "data/raw/isot"))
