"""Thresholding, confidence intervals and model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    val_recall: float
    val_fpr: float


def threshold_for_max_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float) -> OperatingPoint:
    """Highest-recall threshold whose false-positive rate on (y, s) is <= max_fpr."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= max_fpr)[0]
    best = ok[np.argmax(tpr[ok])]
    # first roc_curve threshold is inf
    threshold = float(min(thr[best], 1.0))
    return OperatingPoint(
        threshold=threshold, val_recall=float(tpr[best]), val_fpr=float(fpr[best])
    )


def confusion_at(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, int]:
    pred = s >= threshold
    return {
        "tp": int(np.sum(pred & (y == 1))),
        "fp": int(np.sum(pred & (y == 0))),
        "tn": int(np.sum(~pred & (y == 0))),
        "fn": int(np.sum(~pred & (y == 1))),
    }


def rates(c: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1}


def point_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    c = confusion_at(y, s, threshold)
    out: dict[str, float] = {**rates(c), **{k: float(v) for k, v in c.items()}}
    out["pr_auc"] = float(average_precision_score(y, s))
    out["roc_auc"] = float(roc_auc_score(y, s))
    return out


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for recall, precision, FPR and PR-AUC at a fixed threshold."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    draws: dict[str, list[float]] = {"recall": [], "precision": [], "fpr": [], "pr_auc": []}
    for _ in range(n_resamples):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        yb, sb = y[idx], s[idx]
        r = rates(confusion_at(yb, sb, threshold))
        draws["recall"].append(r["recall"])
        draws["precision"].append(r["precision"])
        draws["fpr"].append(r["fpr"])
        draws["pr_auc"].append(average_precision_score(yb, sb))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
        for k, v in draws.items()
    }


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """McNemar test on paired predictions."""
    a_ok = pred_a == y
    b_ok = pred_b == y
    table = [
        [int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
        [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))],
    ]
    exact = (table[0][1] + table[1][0]) < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    return {
        "a_right_b_wrong": table[0][1],
        "a_wrong_b_right": table[1][0],
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "exact": bool(exact),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index, binned on expected-sample quantiles."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
