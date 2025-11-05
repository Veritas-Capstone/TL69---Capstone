"""Simple baseline training scaffold using scikit-learn.

This module provides a small example of a text baseline: TF-IDF + LogisticRegression.
It's intended as a starting point — tune and extend as needed.
"""
from typing import Sequence, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib


def train_baseline(texts: Sequence[str], labels: Sequence[Any]) -> Tuple[Any, Any]:
    """Train a TF-IDF + LogisticRegression baseline pipeline.

    Returns (pipeline, vectorizer) — pipeline implements fit/predict.
    """
    pipe = make_pipeline(TfidfVectorizer(max_features=20000), LogisticRegression(max_iter=1000))
    pipe.fit(texts, labels)
    return pipe, None


def predict_baseline(pipeline, texts: Sequence[str]):
    return pipeline.predict(texts)


def save_model(pipeline, path: str):
    joblib.dump(pipeline, path)


def load_model(path: str):
    return joblib.load(path)
