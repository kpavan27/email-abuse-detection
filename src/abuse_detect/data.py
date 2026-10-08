"""Download, cleaning, de-duplication and splitting."""

from __future__ import annotations

import hashlib
import io
import logging
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_NONWORD = re.compile(r"\W+")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download(url: str, expected_sha256: str, raw_dir: str | Path) -> Path:
    """Download and unzip the corpus, verifying the archive checksum."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "enron_spam_data.csv"
    if csv_path.exists():
        log.info("raw data already present at %s", csv_path)
        return csv_path
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise ValueError(f"checksum mismatch: expected {expected_sha256}, got {digest}")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(raw_dir)
    return csv_path


def normalise(text: str) -> str:
    """Lowercase, strip digits and collapse whitespace."""
    text = _DIGITS.sub("", text.lower())
    return _WS.sub(" ", text).strip()


def template_key(subject: str, body: str, prefix_chars: int = 120) -> str:
    subj = _NONWORD.sub(" ", subject.lower()).strip()
    return f"{subj}|{normalise(body)[:prefix_chars]}"


@dataclass(frozen=True)
class CleaningReport:
    rows_in: int
    empty_dropped: int
    duplicates_dropped: int
    label_conflicts_dropped: int
    rows_out: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def clean(raw: pd.DataFrame, max_chars: int = 5000) -> tuple[pd.DataFrame, CleaningReport]:
    """Turn the raw CSV into a modelling table."""
    df = pd.DataFrame(
        {
            "message_id": raw["Message ID"].astype(int),
            "subject": raw["Subject"].fillna("").astype(str),
            "body": raw["Message"].fillna("").astype(str),
            "label": (raw["Spam/Ham"].str.lower() == "spam").astype(int),
            "sent_at": pd.to_datetime(raw["Date"], errors="coerce"),
        }
    )
    rows_in = len(df)

    empty = (df["subject"].str.strip() == "") & (df["body"].str.strip() == "")
    df = df.loc[~empty].copy()

    df["text"] = (df["subject"] + "\n" + df["body"]).str.slice(0, max_chars)
    df["norm_hash"] = (
        (df["subject"] + " " + df["body"])
        .map(normalise)
        .map(lambda s: hashlib.md5(s.encode()).hexdigest())
    )

    labels_per_hash = df.groupby("norm_hash")["label"].nunique()
    conflicted = labels_per_hash[labels_per_hash > 1].index
    n_conflict = int(df["norm_hash"].isin(conflicted).sum())
    df = df.loc[~df["norm_hash"].isin(conflicted)]

    before = len(df)
    df = df.drop_duplicates("norm_hash", keep="first")
    n_dupes = before - len(df)

    df["template_key"] = [
        template_key(s, b) for s, b in zip(df["subject"], df["body"], strict=True)
    ]
    df = df.reset_index(drop=True)

    report = CleaningReport(
        rows_in=rows_in,
        empty_dropped=int(empty.sum()),
        duplicates_dropped=int(n_dupes),
        label_conflicts_dropped=n_conflict,
        rows_out=len(df),
    )
    return df, report


def grouped_split(
    df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int
) -> pd.Series:
    """Stratified on label, grouped on template key."""
    groups = pd.factorize(df["template_key"])[0]
    y = df["label"].to_numpy()

    def _carve(idx: np.ndarray, fraction: float, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
        n_splits = max(2, round(1 / fraction))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
        keep, held = next(sgkf.split(idx, y[idx], groups[idx]))
        return idx[keep], idx[held]

    all_idx = np.arange(len(df))
    trainval, test = _carve(all_idx, test_fraction, seed)
    train, val = _carve(trainval, val_fraction, seed + 1)

    split = pd.Series("train", index=df.index, dtype="object")
    split.iloc[val] = "val"
    split.iloc[test] = "test"
    return split


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def mask_tokens(texts: pd.Series, tokens: list[str], strip_years: bool = True) -> pd.Series:
    """Delete identity tokens and 4-digit years."""
    if tokens:
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, tokens)) + r")\b", re.IGNORECASE)
        texts = texts.str.replace(pattern, " ", regex=True)
    if strip_years:
        texts = texts.str.replace(_YEAR, " ", regex=True)
    return texts
