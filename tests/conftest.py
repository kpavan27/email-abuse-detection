import numpy as np
import pandas as pd
import pytest

SPAM = [
    "cheap meds online click here http : / / www . pills . biz",
    "save money now best prices on software remove me",
    "you have won a free prize claim your cash today",
    "v1agra cialis lowest prices order now",
]
HAM = [
    "please review the attached report before the meeting on monday",
    "thanks for the update let me know about the contract schedule",
    "the conference call is moved to friday afternoon",
    "attached are the gas volumes for april please confirm",
]


def make_raw(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def tag(i: int) -> str:  # letters only, digits are normalised away
        return chr(97 + (i // 26) % 26) + chr(97 + i % 26)

    for i in range(n_per_class):
        s = SPAM[i % len(SPAM)] + f" offer {rng.integers(1e6)} ref{i}"
        h = HAM[i % len(HAM)] + f" vince enron item {i} ticket{rng.integers(1e6)}"
        rows.append(
            {
                "Message ID": 2 * i,
                "Subject": f"deal {tag(i)}",
                "Message": s,
                "Spam/Ham": "spam",
                "Date": "2005-03-01",
            }
        )
        rows.append(
            {
                "Message ID": 2 * i + 1,
                "Subject": f"re: item {tag(i)}",
                "Message": h,
                "Spam/Ham": "ham",
                "Date": "2001-03-01",
            }
        )
    # near-duplicate + empty row
    rows.append(
        {
            "Message ID": 9001,
            "Subject": "DEAL AA",
            "Message": rows[0]["Message"].upper() + "   ",
            "Spam/Ham": "spam",
            "Date": "2005-03-02",
        }
    )
    rows.append(
        {
            "Message ID": 9002,
            "Subject": None,
            "Message": None,
            "Spam/Ham": "ham",
            "Date": "2001-01-01",
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()
