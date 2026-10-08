"""Rule-based tagging of misclassified messages."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# corpus is pre-tokenised: "http : / / www . x . biz"
_URL = re.compile(r"https?\s*:\s*/\s*/|www\s*\.", re.I)
# unspaced "@" only, tokenised addresses look like "john @ enron . com"
_OBFUSCATED = re.compile(
    r"[a-z][|~%!][a-z]|[a-z]\s[|~]\s?[a-z]|\b[a-z]+[01][a-z]+\b|[a-z]@[a-z]", re.I
)
_HTML_LEFTOVER = re.compile(r"\b(html|charset|font|nbsp|href)\b", re.I)
_MONEY = re.compile(r"[$£€]\s?\d|\b(free|save|discount|offer|price|cheap|winner|cash)\b", re.I)
_REPLY_FWD = re.compile(r"^\s*(re|fw|fwd)\s*:", re.I)
_BUSINESS = re.compile(
    r"\b(meeting|schedule|attached|please review|conference|report|deal|contract)\b", re.I
)

TAGS = (
    "very_short",
    "link_only",
    "obfuscation",
    "html_leftovers",
    "commercial_language",
    "reply_or_forward",
    "business_vocabulary",
    "mostly_non_ascii",
)


def tag_message(subject: str, body: str) -> list[str]:
    text = f"{subject}\n{body}"
    words = re.findall(r"[A-Za-z]{2,}", body)
    tags: list[str] = []
    if len(body.strip()) < 200:
        tags.append("very_short")
    if _URL.search(body) and len(words) < 40:
        tags.append("link_only")
    if len(_OBFUSCATED.findall(text)) >= 3:
        tags.append("obfuscation")
    if _HTML_LEFTOVER.search(body):
        tags.append("html_leftovers")
    if len(_MONEY.findall(text)) >= 2:
        tags.append("commercial_language")
    if _REPLY_FWD.search(subject):
        tags.append("reply_or_forward")
    if _BUSINESS.search(text):
        tags.append("business_vocabulary")
    non_ascii = sum(ord(ch) > 127 for ch in text)
    if text and non_ascii / len(text) > 0.2:
        tags.append("mostly_non_ascii")
    return tags


def summarise(errors: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Tag rates among errors vs. the same class overall (lift)."""

    def _counts(frame: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for subj, body in zip(frame["subject"], frame["body"], strict=True):
            c.update(tag_message(subj, body))
        return c

    err_c, pop_c = _counts(errors), _counts(population)
    n_err, n_pop = max(len(errors), 1), max(len(population), 1)
    rows = []
    for tag in TAGS:
        share_err = err_c[tag] / n_err
        share_pop = pop_c[tag] / n_pop
        rows.append(
            {
                "tag": tag,
                "errors_with_tag": err_c[tag],
                "share_of_errors": round(share_err, 3),
                "share_of_class": round(share_pop, 3),
                "lift": round(share_err / share_pop, 2) if share_pop else None,
            }
        )
    return pd.DataFrame(rows).sort_values("errors_with_tag", ascending=False).reset_index(drop=True)
