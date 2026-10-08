"""Candidate models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline


def _word_vec() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=200_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_']+\b",
    )


def _char_vec() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=200_000,
        sublinear_tf=True,
    )


def naive_bayes(c: float | None = None) -> Pipeline:
    """Classic spam-filter baseline."""
    return Pipeline([("tfidf", _word_vec()), ("clf", ComplementNB(alpha=0.3))])


def logreg_word(c: float = 2.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", _word_vec()),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="liblinear")),
        ]
    )


def logreg_word_char(c: float = 2.0) -> Pipeline:
    features = FeatureUnion([("word", _word_vec()), ("char", _char_vec())])
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(C=c, max_iter=2000, solver="liblinear")),
        ]
    )


CANDIDATES: dict[str, Callable[..., Pipeline]] = {
    "naive_bayes_word": naive_bayes,
    "logreg_word": logreg_word,
    "logreg_word_char": logreg_word_char,
}

TUNABLE = {"logreg_word", "logreg_word_char"}


def scores(model: Pipeline, texts) -> np.ndarray:
    """Positive-class probability."""
    return model.predict_proba(texts)[:, 1]


def top_features(model: Pipeline, k: int = 25) -> dict[str, list[tuple[str, float]]]:
    """Top spam and ham features of a fitted linear pipeline."""
    clf = model.named_steps["clf"]
    if "features" in model.named_steps:
        names = model.named_steps["features"].get_feature_names_out()
    else:
        names = model.named_steps["tfidf"].get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = clf.coef_.ravel()
    else:  # ComplementNB has no coef_
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(weights)
    ham = [(str(names[i]), float(weights[i])) for i in order[:k]]
    spam = [(str(names[i]), float(weights[i])) for i in order[::-1][:k]]
    return {"spam": spam, "ham": ham}
