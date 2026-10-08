import numpy as np
import pytest

from abuse_detect import evaluate


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000, int), np.ones(1000, int)]
    s = np.r_[rng.beta(2, 8, 1000), rng.beta(8, 2, 1000)]
    return y, s


def test_threshold_respects_fpr_ceiling(scored):
    y, s = scored
    op = evaluate.threshold_for_max_fpr(y, s, max_fpr=0.01)
    r = evaluate.rates(evaluate.confusion_at(y, s, op.threshold))
    assert r["fpr"] <= 0.01 + 1e-12
    assert 0 < op.threshold <= 1


def test_looser_ceiling_never_lowers_recall(scored):
    y, s = scored
    tight = evaluate.threshold_for_max_fpr(y, s, 0.001)
    loose = evaluate.threshold_for_max_fpr(y, s, 0.05)
    assert loose.val_recall >= tight.val_recall


def test_confusion_counts_sum(scored):
    y, s = scored
    c = evaluate.confusion_at(y, s, 0.5)
    assert sum(c.values()) == len(y)


def test_bootstrap_ci_brackets_point_estimate(scored):
    y, s = scored
    point = evaluate.point_metrics(y, s, 0.5)
    ci = evaluate.bootstrap_ci(y, s, 0.5, n_resamples=200, seed=1)
    for key in ("recall", "precision", "fpr", "pr_auc"):
        lo, hi = ci[key]
        assert lo <= point[key] <= hi


def test_mcnemar_identical_predictions_not_significant(scored):
    y, s = scored
    pred = s >= 0.5
    res = evaluate.mcnemar_test(y, pred, pred.copy())
    assert res["a_right_b_wrong"] == res["a_wrong_b_right"] == 0
    assert res["p_value"] == pytest.approx(1.0)


def test_mcnemar_detects_clearly_better_model(scored):
    y, s = scored
    good = s >= 0.5
    bad = np.where(np.arange(len(y)) % 5 == 0, ~good, good)
    assert evaluate.mcnemar_test(y, good, bad)["p_value"] < 1e-6


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 5000)
    assert evaluate.psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert evaluate.psi(a, rng.normal(1.5, 1, 5000)) > 0.25
