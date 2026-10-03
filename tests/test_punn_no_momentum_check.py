from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer_resurrection import punn_no_momentum_check as checker
from optimizer_resurrection.punn_architecture_stability import MANIFOLD_NO_MOMENTUM_METHOD
from optimizer_resurrection.tracking import RunRecord


CONDITION = "registered-condition"
REVISION = "clean-revision"
DIGEST = "clean-source-digest"
EXPECTED = {
    CONDITION: {
        "condition_id": CONDITION,
        "protocol": "punn-manifold-recorded-v1",
        "task": "f1",
        "architecture": "small",
        "method": MANIFOLD_NO_MOMENTUM_METHOD,
        "seed": 0,
        "data_seed": 100000,
        "epochs": 2,
        "momentum": 0.0,
        "learning_rate": 0.01,
        "device": "cpu",
        "landscape_device": "cpu",
    }
}
CONFIG = {
    "protocol": "punn-manifold-recorded-v1",
    "stage": "exploratory",
    "run_group": "no-momentum-test-group",
    "methods": [MANIFOLD_NO_MOMENTUM_METHOD],
    "epochs": 2,
    "slice_every": 1,
}


def _provenance(*, revision=REVISION, digest=DIGEST, nesterov=False, momentum=0.0):
    return {
        "git": {"revision": revision, "dirty": False},
        "source_tree": {"digest": digest},
        "manifold_nesterov": nesterov,
        "optimizer_recipe": {"momentum": momentum},
    }


def _record(*, state="finished", result_changes=None, config_changes=None, run_id="run-a"):
    config = {
        **EXPECTED[CONDITION],
        "stage": "exploratory",
        "run_group": CONFIG["run_group"],
        "provenance": _provenance(),
    }
    if config_changes:
        config.update(config_changes)
    result = {
        "terminal_outcome": "numerical_failure",
        "landscape_recording_complete": True,
        "landscape_artifact": "test-project/landscape:v0",
        "method": MANIFOLD_NO_MOMENTUM_METHOD,
        "momentum": 0.0,
        "manifold_nesterov": False,
    }
    if result_changes:
        result.update(result_changes)
    # W&B exposes selected terminal fields both flat and in the immutable
    # nested snapshot; acceptance requires the flat recording marker.
    summary = {**result, "terminal_result": result}
    if state in {"running", "pending", "preempting"}:
        summary["epoch"] = 1
    return RunRecord(run_id, state, config, summary, f"https://wandb.test/{run_id}")


def _install_fakes(monkeypatch, run_records):
    monkeypatch.setattr(checker, "expected_plan", lambda _config: EXPECTED)
    monkeypatch.setattr(checker, "records", lambda _api, _project, _config: list(run_records))


class _FakeApi:
    def __init__(self, artifact=None):
        self._artifact = artifact
        self.artifact_calls = []

    def artifact(self, reference, type):
        self.artifact_calls.append((reference, type))
        if self._artifact is None:
            raise AssertionError("artifact download should not be reached")
        return self._artifact


def _install_manifest_download(monkeypatch, tmp_path, record):
    result = record.summary["terminal_result"]
    manifest = {
        "condition_id": CONDITION,
        "training_device": "cpu",
        "landscape_device": "cpu",
        "terminal_result": result,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(entries={
            "manifest.json": object(),
            "recorded_landscapes.npz": object(),
            "provenance.json": object(),
        })
    )
    monkeypatch.setattr(checker, "download_entry", lambda _artifact, _name, _directory: manifest_path)
    return artifact


def _check(monkeypatch, records, api=None, *, full=False, cache=Path("cache")):
    _install_fakes(monkeypatch, records)
    return checker.check_group(
        api or _FakeApi(), "test-project", CONFIG, REVISION, DIGEST, cache,
        full=full, engineering=False,
    )


def test_active_condition_cannot_overlap_an_accepted_outcome(monkeypatch):
    accepted = _record()
    active = _record(state="running", run_id="active-run")
    api = _FakeApi()
    with pytest.raises(ValueError, match="active attempt duplicates an accepted"):
        _check(monkeypatch, [active, accepted], api)
    assert api.artifact_calls == []


def test_failed_infrastructure_attempt_stays_excluded_and_condition_is_missing(monkeypatch):
    failed = _record(
        state="failed",
        result_changes={
            "terminal_outcome": "infrastructure_failure",
            "landscape_recording_complete": False,
            "landscape_artifact": None,
        },
    )
    result = _check(monkeypatch, [failed])

    assert result["verified_recorded_outcomes"] == 0
    assert result["missing"] == [CONDITION]
    assert result["excluded"] == [{
        "run_id": "run-a",
        "condition_id": CONDITION,
        "state": "failed",
        "reason": "incomplete_execution_or_recording",
    }]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"provenance": _provenance(revision="wrong-revision")}, "active source mismatch"),
        ({"momentum": 0.95}, "active recipe mismatch"),
    ],
)
def test_active_records_reject_wrong_source_or_registered_recipe(monkeypatch, changes, message):
    active = _record(state="running", config_changes=changes)
    with pytest.raises(ValueError, match=message):
        _check(monkeypatch, [active])


@pytest.mark.parametrize(
    ("result_changes", "config_changes", "message"),
    [
        ({"momentum": 0.95}, None, "terminal recipe mismatch"),
        ({"manifold_nesterov": True}, None, "terminal recipe mismatch"),
        (None, {"provenance": _provenance(nesterov=True)}, "optimizer provenance mismatch"),
        (None, {"provenance": _provenance(momentum=0.95)}, "optimizer provenance mismatch"),
    ],
)
def test_accepted_outcomes_reject_momentum_or_nesterov_mismatches(
    monkeypatch, tmp_path, result_changes, config_changes, message,
):
    record = _record(result_changes=result_changes, config_changes=config_changes)
    artifact = _install_manifest_download(monkeypatch, tmp_path, record)
    api = _FakeApi(artifact)
    with pytest.raises(ValueError, match=message):
        _check(monkeypatch, [record], api)


def test_default_check_reads_manifest_and_full_check_delegates_raw_payload_audit(
    monkeypatch, tmp_path,
):
    record = _record()
    artifact = _install_manifest_download(monkeypatch, tmp_path, record)
    api = _FakeApi(artifact)
    calls = []

    def fake_audit_runs(accepted, project, cache, workers):
        calls.append((accepted, project, cache, workers))
        return [{"run_id": accepted[0].run_id, "raw_payload_verified": True}]

    monkeypatch.setattr(checker, "audit_runs", fake_audit_runs)

    manifest_only = _check(monkeypatch, [record], api, full=False, cache=tmp_path / "manifest")
    assert manifest_only["readback"] == "immutable_manifest"
    assert manifest_only["verified_recorded_outcomes"] == 1
    assert manifest_only["rows"] == []
    assert calls == []

    raw = _check(monkeypatch, [record], api, full=True, cache=tmp_path / "full")
    assert raw["readback"] == "raw_payload"
    assert raw["rows"] == [{"run_id": "run-a", "raw_payload_verified": True}]
    assert len(calls) == 1
    assert calls[0][0] == [record]
    assert calls[0][1] == "test-project"
    assert calls[0][3] == 4
