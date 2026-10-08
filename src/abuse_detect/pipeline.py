"""Experiment pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402

from . import analysis, data, error_analysis, evaluate, models  # noqa: E402

log = logging.getLogger(__name__)


def prepare(cfg: dict[str, Any]) -> pd.DataFrame:
    dcfg, scfg = cfg["data"], cfg["split"]
    csv_path = data.download(dcfg["url"], dcfg["sha256"], dcfg["raw_dir"])
    raw = pd.read_csv(csv_path)
    df, report = data.clean(raw, max_chars=dcfg["max_chars"])
    df["split"] = data.grouped_split(
        df, scfg["test_fraction"], scfg["val_fraction"], scfg["seed"]
    ).values

    out = Path(dcfg["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "cleaning": report.as_dict(),
        "split_sizes": df.groupby(["split", "label"])
        .size()
        .unstack()
        .rename(columns={0: "ham", 1: "spam"})
        .to_dict(orient="index"),
        "template_groups_crossing_splits": int(
            (df.groupby("template_key")["split"].nunique() > 1).sum()
        ),
    }
    (reports / "data_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    log.info("prepared %d rows: %s", len(df), summary["cleaning"])
    return df


def _mask_fn(texts, tokens: list[str]):
    return data.mask_tokens(pd.Series(list(texts), dtype="object"), tokens).tolist()


def with_masking(model: Pipeline, tokens: list[str]) -> Pipeline:
    """Prepend masking so the saved model applies it at serving time."""
    return Pipeline(
        [("mask", FunctionTransformer(partial(_mask_fn, tokens=tokens))), ("model", model)]
    )


def _fit_with_c_search(
    name: str,
    x_tr: list[str],
    y_tr: np.ndarray,
    x_va: list[str],
    y_va: np.ndarray,
    c_grid: list[float],
) -> tuple[Pipeline, dict[str, float]]:
    """Vectorise once, then search C."""
    template = models.CANDIDATES[name]()
    if name not in models.TUNABLE:
        fitted = clone(template).fit(x_tr, y_tr)
        return fitted, {"none": float(average_precision_score(y_va, models.scores(fitted, x_va)))}

    feat_name = "features" if "features" in template.named_steps else "tfidf"
    vec = clone(template.named_steps[feat_name]).fit(x_tr)
    m_tr, m_va = vec.transform(x_tr), vec.transform(x_va)
    search: dict[str, float] = {}
    best_c, best_ap, best_clf = None, -1.0, None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=2000, solver="liblinear").fit(m_tr, y_tr)
        ap = float(average_precision_score(y_va, clf.predict_proba(m_va)[:, 1]))
        search[str(c)] = ap
        if ap > best_ap:
            best_c, best_ap, best_clf = c, ap, clf
    log.info("%s: best C=%s val PR-AUC=%.4f", name, best_c, best_ap)
    return Pipeline([(feat_name, vec), ("clf", best_clf)]), search


@dataclass
class Splits:
    tr: pd.DataFrame
    va: pd.DataFrame
    te: pd.DataFrame
    x_tr: list[str]
    x_va: list[str]
    x_te: list[str]

    @property
    def y_tr(self) -> np.ndarray:
        return self.tr["label"].to_numpy()

    @property
    def y_va(self) -> np.ndarray:
        return self.va["label"].to_numpy()

    @property
    def y_te(self) -> np.ndarray:
        return self.te["label"].to_numpy()


def make_splits(df: pd.DataFrame, tokens: list[str]) -> Splits:
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    mask = partial(_mask_fn, tokens=tokens)
    return Splits(tr, va, te, mask(tr.text), mask(va.text), mask(te.text))


@dataclass
class Selection:
    best: str
    threshold: float
    fitted: dict[str, Pipeline]
    test_scores: dict[str, np.ndarray]
    report: dict[str, Any]


def select_model(sp: Splits, cfg: dict[str, Any], baseline: str = "naive_bayes_word") -> Selection:
    """Pick by validation PR-AUC, fix the threshold on validation, evaluate once on test."""
    max_fpr = cfg["operating_point"]["max_fpr"]
    report: dict[str, Any] = {"candidates": {}}
    fitted: dict[str, Pipeline] = {}
    test_scores: dict[str, np.ndarray] = {}
    for name in models.CANDIDATES:
        model, search = _fit_with_c_search(
            name, sp.x_tr, sp.y_tr, sp.x_va, sp.y_va, cfg["models"]["c_grid"]
        )
        op = evaluate.threshold_for_max_fpr(sp.y_va, models.scores(model, sp.x_va), max_fpr)
        fitted[name] = model
        test_scores[name] = models.scores(model, sp.x_te)
        report["candidates"][name] = {
            "val_pr_auc_by_c": search,
            "threshold": op.threshold,
            "val_recall_at_threshold": op.val_recall,
            "test": evaluate.point_metrics(sp.y_te, test_scores[name], op.threshold),
        }

    cands = report["candidates"]
    best = max(cands, key=lambda n: max(cands[n]["val_pr_auc_by_c"].values()))
    thr = cands[best]["threshold"]
    ecfg = cfg["evaluation"]
    report["selected_model"] = best
    report["selected_test_ci95"] = evaluate.bootstrap_ci(
        sp.y_te, test_scores[best], thr, ecfg["bootstrap_resamples"], ecfg["seed"]
    )
    report["mcnemar_selected_vs_baseline"] = evaluate.mcnemar_test(
        sp.y_te,
        test_scores[best] >= thr,
        test_scores[baseline] >= cands[baseline]["threshold"],
    )
    report["top_features"] = models.top_features(fitted[best], k=20)
    return Selection(best, thr, fitted, test_scores, report)


def shortcut_experiment(sp: Splits, recipe: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Same model trained on unmasked text."""
    raw_va = sp.va.text.tolist()
    model, _ = _fit_with_c_search(
        recipe, sp.tr.text.tolist(), sp.y_tr, raw_va, sp.y_va, cfg["models"]["c_grid"]
    )
    thr = evaluate.threshold_for_max_fpr(
        sp.y_va, models.scores(model, raw_va), cfg["operating_point"]["max_fpr"]
    ).threshold
    return {
        "description": "Model trained on unmasked text, evaluated on unmasked test and on the "
        "same test messages with identity tokens removed "
        "(simulating mail from outside the Enron mailboxes).",
        "unmasked_test": evaluate.point_metrics(
            sp.y_te, models.scores(model, sp.te.text.tolist()), thr
        ),
        "identity_removed_test": evaluate.point_metrics(
            sp.y_te, models.scores(model, sp.x_te), thr
        ),
        "top_ham_features_unmasked": models.top_features(model, k=15)["ham"],
    }


def drift_experiment(
    sp: Splits, recipe: str, cfg: dict[str, Any], cutoff: str = "2005-01-01"
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    """Train on spam before ``cutoff``, compare recall on older vs newer spam."""
    ts = pd.Timestamp(cutoff)
    keep_tr = ((sp.tr.label == 0) | (sp.tr.sent_at < ts)).to_numpy()
    keep_va = ((sp.va.label == 0) | (sp.va.sent_at < ts)).to_numpy()
    x_tr = [x for x, k in zip(sp.x_tr, keep_tr, strict=True) if k]
    x_va = [x for x, k in zip(sp.x_va, keep_va, strict=True) if k]
    model, _ = _fit_with_c_search(
        recipe, x_tr, sp.y_tr[keep_tr], x_va, sp.y_va[keep_va], cfg["models"]["c_grid"]
    )
    thr = evaluate.threshold_for_max_fpr(
        sp.y_va[keep_va], models.scores(model, x_va), cfg["operating_point"]["max_fpr"]
    ).threshold
    s = models.scores(model, sp.x_te)
    old = ((sp.te.label == 1) & (sp.te.sent_at < ts)).to_numpy()
    new = ((sp.te.label == 1) & (sp.te.sent_at >= ts)).to_numpy()
    report = {
        "description": "Trained on ham + spam sent before 2005; recall compared on held-out "
        "spam from before 2005 (in-period) and from 2005 (future).",
        "in_period_spam_n": int(old.sum()),
        "future_spam_n": int(new.sum()),
        "in_period_recall": float(np.mean(s[old] >= thr)),
        "future_recall": float(np.mean(s[new] >= thr)),
        "score_psi_in_period_vs_future": evaluate.psi(s[old], s[new]),
    }
    return report, s[old], s[new], thr


def error_review(sp: Splits, scores: np.ndarray, thr: float, tables: Path) -> dict[str, Any]:
    pred = scores >= thr
    fn = sp.te[(sp.y_te == 1) & ~pred]
    fp = sp.te[(sp.y_te == 0) & pred]
    fn_tbl = error_analysis.summarise(fn, sp.te[sp.te.label == 1])
    fp_tbl = error_analysis.summarise(fp, sp.te[sp.te.label == 0])
    fn_tbl.to_csv(tables / "false_negative_tags.csv", index=False)
    fp_tbl.to_csv(tables / "false_positive_tags.csv", index=False)
    return {
        "false_negatives": int(len(fn)),
        "false_positives": int(len(fp)),
        "fn_tags": fn_tbl.to_dict(orient="records"),
        "fp_tags": fp_tbl.to_dict(orient="records"),
        "fp_subjects": fp["subject"].str.slice(0, 80).tolist()[:15],
    }


def package_model(sel: Selection, sp: Splits, cfg: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        with_masking(sel.fitted[sel.best], cfg["identity_tokens"]),
        artifacts / "model.joblib",
        compress=3,
    )
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    metadata = {
        "model_name": sel.best,
        "model_version": f"{sel.best}-{cfg_hash}",
        "threshold": sel.threshold,
        "max_fpr_target": cfg["operating_point"]["max_fpr"],
        "data_sha256": cfg["data"]["sha256"],
        "config_hash": cfg_hash,
        "train_rows": int(len(sp.tr)),
        "test_metrics": sel.report["candidates"][sel.best]["test"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    reports = Path(cfg["paths"]["reports"])
    figs, tables = reports / "figures", reports / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    analysis.run_all(cfg["data"]["processed_path"], tables)
    _plot_volume(tables / "01_monthly_volume.csv", figs / "monthly_volume.png")

    sp = make_splits(pd.read_parquet(cfg["data"]["processed_path"]), cfg["identity_tokens"])
    sel = select_model(sp, cfg)
    results = dict(sel.report)
    _plot_pr(
        sp.y_te, sel.test_scores, figs / "recall_vs_fpr_test.png", cfg["operating_point"]["max_fpr"]
    )
    pd.DataFrame({"message_id": sp.te.message_id, "label": sp.y_te, **sel.test_scores}).to_parquet(
        reports / "test_scores.parquet", index=False
    )

    results["shortcut_experiment"] = shortcut_experiment(sp, sel.best, cfg)
    drift, old, new, d_thr = drift_experiment(sp, sel.best, cfg)
    results["drift_experiment"] = drift
    _plot_drift(old, new, d_thr, figs / "drift_score_distribution.png")
    results["error_analysis"] = error_review(sp, sel.test_scores[sel.best], sel.threshold, tables)
    package_model(sel, sp, cfg)

    results["runtime_seconds"] = round(time.time() - t0, 1)
    (reports / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    log.info("done in %.0fs; selected %s", results["runtime_seconds"], sel.best)
    return results


def _plot_pr(y: np.ndarray, scores: dict[str, np.ndarray], path: Path, max_fpr: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.step(np.clip(fpr, 1e-4, 1), tpr, where="post", label=name)
    ax.axvline(max_fpr, color="k", ls="--", lw=1, label=f"FPR budget {max_fpr:.1%}")
    ax.set_xscale("log")
    ax.set(
        xlim=(3e-4, 0.05),
        ylim=(0.8, 1.0),
        xlabel="False-positive rate (legitimate mail blocked), log scale",
        ylabel="Recall (spam caught)",
        title="Recall at low false-positive rates, held-out test",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_drift(old: np.ndarray, new: np.ndarray, thr: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(old, bins=bins, alpha=0.6, density=True, label="spam before 2005 (in period)")
    ax.hist(new, bins=bins, alpha=0.6, density=True, label="spam from 2005 (future)")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"threshold {thr:.2f}")
    ax.set_yscale("log")
    ax.set(
        xlabel="Spam probability",
        ylabel="Density (log scale)",
        title="Newer spam scores lower: mass moves below the threshold",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_volume(csv_path: Path, path: Path) -> None:
    vol = pd.read_csv(csv_path, parse_dates=["month"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(vol["month"], vol["ham"], width=25, label="ham")
    ax.bar(vol["month"], vol["spam"], width=25, bottom=vol["ham"], label="spam")
    ax.set(
        title="Messages per month: the two classes come from different periods", ylabel="Messages"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
