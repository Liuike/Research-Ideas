from copy import deepcopy

import pytest

from optimizer_resurrection.punn_manifold_resume import (
    pending_condition_ids,
    pending_continuation_condition_ids,
)
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


GPU_GROUP = "punn-manifold-landscape-gpu-v1"
CPU_GROUP = "punn-manifold-landscape-cpu-continuation-v1"
GPU_REVISION = "c55600e4755adf687753e6dee01b63462bbc6f65"
GPU_DIGEST = "853afa71ba5f0e4df8f1a78e4967d573c5f476e6ef35522c176e5639768946c6"
CPU_REVISION = "cpu-revision"
CPU_DIGEST = "cpu-digest"


def _continuation_expected(condition, device):
    return {
        "condition_id": condition,
        "protocol": "punn-manifold-recorded-v1",
        "task": "f1",
        "architecture": "small",
        "method": "manifold_muon_da10",
        "seed": 0,
        "data_seed": 100000,
        "epochs": 500,
        "regularization_lambda": 0.0001,
        "learning_rate": 0.01,
        "slice_points": 31,
        "slice_radius": 1.0,
        "landscape_views": ["ambient", "manifold"],
        "device": device,
        "landscape_device": device,
    }


GPU_CONTINUATION_EXPECTED = {"gpu-a": _continuation_expected("gpu-a", "cuda"),
                            "gpu-b": {**_continuation_expected("gpu-b", "cuda"), "seed": 1}}
CPU_CONTINUATION_EXPECTED = {"cpu-a": _continuation_expected("cpu-a", "cpu"),
                            "cpu-b": {**_continuation_expected("cpu-b", "cpu"), "seed": 1}}


def _continuation_record(condition, expected, *, device, group, run_id,
                         state="finished", complete=True, outcome="completed",
                         revision=None, digest=None):
    revision = revision or (GPU_REVISION if device == "cuda" else CPU_REVISION)
    digest = digest or (GPU_DIGEST if device == "cuda" else CPU_DIGEST)
    config = {
        **expected,
        "run_group": group,
        "stage": "exploratory",
        "provenance": {
            "git": {"revision": revision, "dirty": False},
            "source_tree": {"digest": digest},
        },
    }
    return RunRecord(
        run_id, state, config,
        {"terminal_result": {
            "terminal_outcome": outcome,
            "epochs_completed": expected["epochs"] if outcome == "completed" else 0,
            "landscape_recording_complete": complete,
            "landscape_artifact": f"project/{run_id}:v0" if complete else None,
        }},
        f"https://wandb.test/{run_id}",
    )


def _reconcile(source_records=(), current_records=(),
               source_expected=GPU_CONTINUATION_EXPECTED,
               current_expected=CPU_CONTINUATION_EXPECTED):
    return pending_continuation_condition_ids(
        source_expected, current_expected, list(source_records), list(current_records),
        source_group=GPU_GROUP, current_group=CPU_GROUP,
        source_revision=GPU_REVISION, source_tree_digest=GPU_DIGEST,
        current_revision=CPU_REVISION, current_tree_digest=CPU_DIGEST,
    )


def test_continuation_skips_unique_uploaded_gpu_and_cpu_outcomes():
    gpu = _continuation_record("gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"],
                               device="cuda", group=GPU_GROUP, run_id="gpu-run-a")
    cpu = _continuation_record("cpu-b", CPU_CONTINUATION_EXPECTED["cpu-b"],
                               device="cpu", group=CPU_GROUP, run_id="cpu-run-b")
    pending, excluded, accepted_sources = _reconcile([gpu], [cpu])
    assert pending == set()
    assert excluded == []
    assert accepted_sources == [{
        "run_id": "gpu-run-a", "condition_id": "gpu-a", "device": "cuda",
        "landscape_artifact": "project/gpu-run-a:v0",
    }]


def test_continuation_rejects_one_cell_completed_in_both_groups():
    gpu = _continuation_record("gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"],
                               device="cuda", group=GPU_GROUP, run_id="gpu-run-a")
    cpu = _continuation_record("cpu-a", CPU_CONTINUATION_EXPECTED["cpu-a"],
                               device="cpu", group=CPU_GROUP, run_id="cpu-run-a")
    with pytest.raises(ValueError, match="duplicate accepted GPU and CPU"):
        _reconcile([gpu], [cpu])


def test_continuation_retries_interrupted_gpu_attempt_and_preserves_exclusion():
    interrupted = _continuation_record(
        "gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"], device="cuda", group=GPU_GROUP,
        run_id="gpu-interrupted", state="failed", complete=False,
        outcome="infrastructure_failure",
    )
    pending, excluded, accepted_sources = _reconcile([interrupted])
    assert pending == {"cpu-a", "cpu-b"}
    assert accepted_sources == []
    assert excluded == [{
        "run_id": "gpu-interrupted", "condition_id": "gpu-a", "state": "failed",
        "reason": "incomplete_execution_or_recording", "device": "cuda", "group": GPU_GROUP,
    }]


def test_continuation_rejects_duplicate_accepted_cells_and_active_attempts():
    gpu = _continuation_record("gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"],
                               device="cuda", group=GPU_GROUP, run_id="gpu-run-a")
    duplicate = _continuation_record("gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"],
                                     device="cuda", group=GPU_GROUP, run_id="gpu-run-a2")
    with pytest.raises(ValueError, match="duplicate accepted"):
        _reconcile([gpu, duplicate])
    active = _continuation_record("cpu-b", CPU_CONTINUATION_EXPECTED["cpu-b"],
                                  device="cpu", group=CPU_GROUP, run_id="cpu-active",
                                  state="running", complete=False)
    with pytest.raises(ValueError, match="active landscape"):
        _reconcile(current_records=[active])


def test_continuation_rejects_duplicate_expected_logical_cells():
    source_expected = deepcopy(GPU_CONTINUATION_EXPECTED)
    source_expected["gpu-alias"] = {**source_expected["gpu-a"], "condition_id": "gpu-alias"}
    with pytest.raises(ValueError, match="duplicate source expected logical cell"):
        _reconcile(source_expected=source_expected)


@pytest.mark.parametrize("change,match", [
    ("recipe", "continuation recipe mismatch"),
    ("source_bytes", "original source bytes"),
    ("source_revision", "original clean source"),
    ("group", "unknown landscape run group"),
    ("device", "landscape device mismatch"),
    ("condition", "unregistered landscape condition"),
    ("slice", "landscape recipe mismatch"),
])
def test_continuation_rejects_recipe_identity_and_provenance_mismatches(change, match):
    source = _continuation_record("gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"],
                                  device="cuda", group=GPU_GROUP, run_id="gpu-run-a")
    source_expected = deepcopy(GPU_CONTINUATION_EXPECTED)
    if change == "recipe":
        current_expected = deepcopy(CPU_CONTINUATION_EXPECTED)
        current_expected["cpu-a"]["slice_radius"] = 2.0
        with pytest.raises(ValueError, match=match):
            _reconcile([source], source_expected=source_expected,
                       current_expected=current_expected)
        return
    if change == "source_bytes":
        source.config["provenance"]["source_tree"]["digest"] = "wrong"
    elif change == "source_revision":
        source.config["provenance"]["git"]["revision"] = "wrong"
    elif change == "group":
        source.config["run_group"] = "unregistered-group"
    elif change == "device":
        source.config["device"] = "cpu"
    elif change == "condition":
        source.config["condition_id"] = "unknown"
    elif change == "slice":
        source.config["slice_points"] = 29
    with pytest.raises(ValueError, match=match):
        _reconcile([source])


def test_continuation_requires_current_cpu_runs_to_match_clean_current_source():
    current = _continuation_record("cpu-a", CPU_CONTINUATION_EXPECTED["cpu-a"],
                                   device="cpu", group=CPU_GROUP, run_id="cpu-run-a")
    current.config["provenance"]["source_tree"]["digest"] = "stale"
    with pytest.raises(ValueError, match="original source bytes"):
        _reconcile(current_records=[current])


def test_continuation_rejects_engineering_stage_records():
    source = _continuation_record("gpu-a", GPU_CONTINUATION_EXPECTED["gpu-a"],
                                  device="cuda", group=GPU_GROUP, run_id="gpu-run-a")
    source.config["stage"] = "engineering-smoke"
    with pytest.raises(ValueError, match="scientific exploratory stage"):
        _reconcile([source])
