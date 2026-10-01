from copy import deepcopy

import pytest

from optimizer_resurrection.punn_manifold_resume import pending_condition_ids
from optimizer_resurrection.tracking import RunRecord


def _record(state="finished", complete=True):
    return RunRecord("run-1", state,
                     {"condition_id": "a", "epochs": 500, "device": "cuda",
                      "provenance": {"git": {"revision": "source", "dirty": False}}},
                     {"terminal_result": {"terminal_outcome": "completed", "epochs_completed": 500,
                                          "landscape_recording_complete": complete,
                                          "landscape_artifact": "project/artifact:v0"}}, "url")


EXPECTED = {"a": {"condition_id": "a", "epochs": 500, "device": "cuda"},
            "b": {"condition_id": "b", "epochs": 500, "device": "cuda"}}


def test_resume_skips_uploaded_outcomes_and_discloses_incomplete_attempts():
    complete = _record()
    pending, excluded = pending_condition_ids(EXPECTED, [complete], "source")
    assert pending == {"b"}
    assert excluded == []
    incomplete = _record(complete=False)
    pending, excluded = pending_condition_ids(EXPECTED, [incomplete], "source")
    assert pending == {"a", "b"}
    assert excluded[0]["reason"] == "incomplete_execution_or_recording"


def test_resume_never_launches_over_an_active_run_or_duplicate_outcome():
    with pytest.raises(ValueError, match="active landscape"):
        pending_condition_ids(EXPECTED, [_record(state="running")], "source")
    with pytest.raises(ValueError, match="duplicate accepted"):
        pending_condition_ids(EXPECTED, [_record(), _record()], "source")


def test_resume_rejects_different_source_bytes_even_at_same_git_revision():
    record = _record()
    record.config["provenance"]["source_tree"] = {"digest": "different"}
    with pytest.raises(ValueError, match="original source bytes"):
        pending_condition_ids(EXPECTED, [record], "source", "expected")


@pytest.mark.parametrize("change,match", [
    ("device", "recipe mismatch"), ("dirty", "original clean source"),
    ("epochs", "incomplete final epoch"), ("artifact", "immutable landscape"),
])
def test_resume_rejects_wrong_recipe_source_and_incomplete_recording(change, match):
    record = deepcopy(_record())
    if change == "device":
        record.config["device"] = "cpu"
    elif change == "dirty":
        record.config["provenance"]["git"]["dirty"] = True
    elif change == "epochs":
        record.summary["terminal_result"]["epochs_completed"] = 400
    else:
        record.summary["terminal_result"].pop("landscape_artifact")
    with pytest.raises(ValueError, match=match):
        pending_condition_ids(EXPECTED, [record], "source")
