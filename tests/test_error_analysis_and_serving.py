import json

import pandas as pd
import pytest

from abuse_detect import error_analysis, pipeline
from abuse_detect.models import logreg_word_char


def test_tags_detect_expected_patterns():
    assert "very_short" in error_analysis.tag_message("hi", "short")
    assert "link_only" in error_analysis.tag_message("x", "go to http : / / www . site . biz now")
    assert "reply_or_forward" in error_analysis.tag_message("RE: schedule", "see below " * 30)
    assert "obfuscation" in error_analysis.tag_message("", "v1agra and c1alis and ph@rmacy deals")


def test_tokenised_email_address_is_not_obfuscation():
    body = "contact john @ enron . com or sally @ enron . com or mark @ enron . com " * 3
    assert "obfuscation" not in error_analysis.tag_message("", body)


def test_summary_lift():
    pop = pd.DataFrame({"subject": ["a"] * 10, "body": ["x" * 300] * 8 + ["tiny"] * 2})
    err = pop.tail(2)
    tbl = error_analysis.summarise(err, pop).set_index("tag")
    assert tbl.loc["very_short", "share_of_errors"] == 1.0
    assert tbl.loc["very_short", "lift"] == pytest.approx(5.0)


@pytest.fixture
def artifact_dir(tmp_path, raw_df):
    texts = (raw_df["Subject"].fillna("") + "\n" + raw_df["Message"].fillna("")).tolist()
    y = (raw_df["Spam/Ham"] == "spam").astype(int).to_numpy()
    model = pipeline.with_masking(logreg_word_char(2.0).fit(texts, y), ["enron", "vince"])
    import joblib

    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"threshold": 0.5, "model_version": "test-1"})
    )
    return tmp_path


def test_api_scores_and_validates(artifact_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from abuse_detect import serve

    monkeypatch.setattr(serve, "ARTIFACT_DIR", artifact_dir)
    serve._load.cache_clear()
    client = TestClient(serve.app)

    assert client.get("/healthz").json()["model_version"] == "test-1"

    spam = client.post(
        "/v1/score", json={"subject": "deal", "body": "cheap meds online click here save money"}
    )
    ham = client.post(
        "/v1/score", json={"subject": "re: item", "body": "please review the attached report"}
    )
    assert spam.status_code == ham.status_code == 200
    assert spam.json()["spam_probability"] > ham.json()["spam_probability"]
    assert spam.json()["decision"] in {"block", "allow"}

    assert client.post("/v1/score", json={"subject": "x"}).status_code == 422


def test_masking_is_inside_the_artifact(artifact_dir):
    import joblib

    model = joblib.load(artifact_dir / "model.joblib")
    assert model.steps[0][0] == "mask"
