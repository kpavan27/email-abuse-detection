import pandas as pd

from abuse_detect import data


def test_normalise_ignores_case_digits_whitespace():
    assert data.normalise("Hello  World 123\n") == data.normalise("hello world 9")


def test_clean_drops_empty_and_duplicates(raw_df):
    df, report = data.clean(raw_df)
    assert report.empty_dropped == 1
    assert report.duplicates_dropped >= 1
    assert report.rows_out == len(df)
    assert df["norm_hash"].is_unique
    assert set(df["label"]) == {0, 1}


def test_clean_drops_label_conflicts(raw_df):
    conflict = raw_df.iloc[[0]].copy()
    conflict["Spam/Ham"] = "ham"
    conflict["Message ID"] = 12345
    df, report = data.clean(pd.concat([raw_df, conflict], ignore_index=True))
    assert report.label_conflicts_dropped >= 2
    assert not df["norm_hash"].duplicated().any()


def test_grouped_split_keeps_templates_together(raw_df):
    df, _ = data.clean(raw_df)
    split = data.grouped_split(df, test_fraction=0.2, val_fraction=0.25, seed=1)
    assert set(split.unique()) == {"train", "val", "test"}
    crossing = df.assign(split=split.values).groupby("template_key")["split"].nunique()
    assert (crossing == 1).all()


def test_grouped_split_is_deterministic(raw_df):
    df, _ = data.clean(raw_df)
    a = data.grouped_split(df, 0.2, 0.25, seed=3)
    b = data.grouped_split(df, 0.2, 0.25, seed=3)
    assert a.equals(b)


def test_mask_tokens_deletes_rather_than_replaces():
    out = data.mask_tokens(pd.Series(["Vince at ENRON wrote in 2005"]), ["enron", "vince"])[0]
    assert "enron" not in out.lower() and "vince" not in out.lower()
    assert "2005" not in out
    assert "_id_" not in out  # placeholder became a label proxy
    assert "enronx" in data.mask_tokens(pd.Series(["enronx"]), ["enron"])[0]
