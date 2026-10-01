from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.nn.utils import parameters_to_vector

from optimizer_resurrection import punn_manifold_recorded as recorded
from optimizer_resurrection import punn_architecture_stability as stability
from optimizer_resurrection.optim.manifold_muon import ManifoldMuon
from optimizer_resurrection.optim.manifold_muon import retract_stiefel


ROOT = Path(__file__).resolve().parents[1]
CPU_SMOKE_CONFIG = ROOT / "configs/product_unit/manifold_landscape_cpu_smoke.yaml"


class _FakeArtifact:
    def __init__(self, name, artifact_type, description, metadata):
        self.name = name
        self.type = artifact_type
        self.description = description
        self.metadata = metadata
        self.qualified_name = f"test-entity/test-project/{name}:v0"
        self.manifest = SimpleNamespace(entries={})
        self.files = {}

    @contextmanager
    def new_file(self, name, mode="w", encoding=None):
        stream = io.BytesIO() if "b" in mode else io.StringIO()
        try:
            yield stream
        finally:
            self.files[name] = stream.getvalue()
            self.manifest.entries[name] = {"name": name}


class _FakeLoggedArtifact:
    def __init__(self, artifact, events):
        self.name = artifact.name
        self.qualified_name = artifact.qualified_name
        self.manifest = artifact.manifest
        self._artifact = artifact
        self._events = events

    def wait(self):
        self._events.append(("artifact_wait", self._artifact.type, self.name))
        return self


class _FakeSummary(dict):
    def __init__(self, events):
        super().__init__()
        self._events = events

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        if values.get("landscape_recording_complete") is True:
            self._events.append(("recording_marker", True))
        return super().update(values)


class _FakeConfig(dict):
    def update(self, *args, **kwargs):
        kwargs.pop("allow_val_change", None)
        return super().update(*args, **kwargs)


class _FakeRun:
    def __init__(self, run_id, init_kwargs, events):
        self.id = run_id
        self.url = f"https://wandb.invalid/{run_id}"
        self.init_kwargs = init_kwargs
        self.group = init_kwargs["group"]
        self.config = _FakeConfig(init_kwargs["config"])
        self.summary = _FakeSummary(events)
        self._events = events
        self.finish_codes = []
        self.artifacts = []

    def log(self, *args, **kwargs):
        return None

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)
        return _FakeLoggedArtifact(artifact, self._events)

    def finish(self, exit_code=0):
        self.finish_codes.append(exit_code)
        self._events.append(("finish", exit_code))


class _FakeWandb:
    def __init__(self, events):
        self.events = events
        self.runs = []
        self.artifacts = []

    def init(self, **kwargs):
        run = _FakeRun(f"offline-memory-{len(self.runs)}", kwargs, self.events)
        self.runs.append(run)
        return run

    def Artifact(self, name, type, description, metadata):
        artifact = _FakeArtifact(name, type, description, metadata)
        self.artifacts.append(artifact)
        return artifact


def _cpu_smoke_argv(architecture: str) -> list[str]:
    config = yaml.safe_load(CPU_SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["conditions"] = [{"task": "xor", "architecture": architecture}]
    config["seeds"] = [config["seeds"][0]]
    command = recorded.expand_config(config)[0]
    return command[3:]


def _install_mocks(monkeypatch, events):
    fake_wandb = _FakeWandb(events)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        recorded,
        "load_wandb_credentials",
        lambda _root: {
            "WANDB_API_KEY": "memory-only-key",
            "WANDB_ENTITY": "test-entity",
            "WANDB_PROJECT": "test-project",
        },
    )
    monkeypatch.setattr(recorded, "require_online_wandb", lambda: None)
    monkeypatch.setattr(
        recorded,
        "_git_metadata",
        lambda _root: {"revision": "engineering-smoke-revision", "dirty": False},
    )
    monkeypatch.setattr(
        recorded,
        "_source_tree_digest",
        lambda _root: {"algorithm": "sha256", "digest": "test-digest", "file_count": 1},
    )
    monkeypatch.setattr(
        recorded, "_environment_metadata", lambda device: {"device": str(device)}
    )
    monkeypatch.setattr(
        recorded,
        "_determinism_metadata",
        lambda seed, data_seed: {"seed": seed, "data_seed": data_seed},
    )
    monkeypatch.setattr(
        recorded, "_sanitized_invocation", lambda _invocation, _credentials: ["<redacted>"]
    )
    return fake_wandb


def _instrument_training(monkeypatch, *, inject_svd_failure=False):
    snapshots = []
    step_calls = []
    previous_condition_ids = []
    actual_run_trial = stability.run_trial
    actual_step = ManifoldMuon.step

    def traced_step(self, *args, **kwargs):
        step_calls.append(type(self))
        result = actual_step(self, *args, **kwargs)
        if inject_svd_failure and len(step_calls) == 1:
            self.param_groups[0]["params"][0].data.zero_()
            raise torch.linalg.LinAlgError("forced finite DA-10 SVD failure")
        return result

    def traced_run_trial(args, *call_args, **kwargs):
        previous_condition_ids.append(
            stability._condition_id(stability._scientific_values(args))
        )
        original_observer = kwargs["on_model_state"]

        def capture(model, metadata):
            if metadata["phase"] in {"initialization", "terminal"}:
                snapshots.append(
                    (metadata["phase"], model.exponents.detach().cpu().clone())
                )
            original_observer(model, metadata)

        kwargs["on_model_state"] = capture
        return actual_run_trial(args, *call_args, **kwargs)

    monkeypatch.setattr(ManifoldMuon, "step", traced_step)
    monkeypatch.setattr(stability, "run_trial", traced_run_trial)
    return snapshots, step_calls, previous_condition_ids


@pytest.mark.parametrize("architecture", ["small", "oversized"])
def test_main_records_real_da10_trajectory_and_both_landscape_views(
    monkeypatch, architecture, capsys
):
    events = []
    fake_wandb = _install_mocks(monkeypatch, events)
    snapshots, step_calls, old_condition_ids = _instrument_training(monkeypatch)

    result = recorded.main(_cpu_smoke_argv(architecture))
    capsys.readouterr()

    assert result["terminal_outcome"] == "completed"
    assert result["condition_id"] != old_condition_ids[0]
    assert step_calls and all(kind is ManifoldMuon for kind in step_calls)
    assert [phase for phase, _ in snapshots] == ["initialization", "terminal"]
    for _, exponents in snapshots:
        assert torch.allclose(
            exponents, retract_stiefel(exponents, scale=1.0), atol=1e-5, rtol=1e-5
        )

    assert result["recorded_state_count"] == 4
    assert result["parameter_states_count"] == 4
    assert result["recorded_slice_count"] == 8
    assert result["slice_count_by_view"] == {"ambient": 4, "manifold": 4}
    assert result["landscape_recording_complete"] is True

    run = fake_wandb.runs[0]
    assert run.init_kwargs["mode"] == "online"
    assert run.init_kwargs["save_code"] is False
    assert run.config["stage"] == "engineering-smoke"
    assert run.config["protocol"] == recorded.PROTOCOL
    assert run.config["provenance"]["git"] == {
        "revision": "engineering-smoke-revision",
        "dirty": False,
    }
    assert run.config["provenance"]["training_protocol"] == stability.MANIFOLD_PROTOCOL
    assert run.config["provenance"]["precision"].startswith("CPU FP32 scalar batch-one")
    assert run.summary["terminal_result"]["condition_id"] == result["condition_id"]
    assert run.finish_codes == [0]

    landscape_wait = next(
        index for index, event in enumerate(events)
        if event[:2] == ("artifact_wait", "loss-landscape")
    )
    marker = next(
        index for index, event in enumerate(events) if event == ("recording_marker", True)
    )
    assert landscape_wait < marker
    landscape_artifact = next(
        artifact for artifact in fake_wandb.artifacts if artifact.type == "loss-landscape"
    )
    assert {"dataset_metadata.npz", "recorded_landscapes.npz", "provenance.json", "manifest.json"}.issubset(
        landscape_artifact.files
    )


def test_accepted_finite_da10_svd_failure_finishes_successfully(monkeypatch, capsys):
    events = []
    fake_wandb = _install_mocks(monkeypatch, events)
    snapshots, step_calls, old_condition_ids = _instrument_training(
        monkeypatch, inject_svd_failure=True
    )

    result = recorded.main(_cpu_smoke_argv("small"))
    capsys.readouterr()

    assert result["terminal_outcome"] == "numerical_failure"
    assert result["failure_reason"] == "manifold_svd_failure"
    assert result["condition_id"] != old_condition_ids[0]
    assert step_calls and all(kind is ManifoldMuon for kind in step_calls)
    assert [phase for phase, _ in snapshots] == ["initialization", "terminal"]
    assert result["recorded_state_count"] == 2
    assert result["recorded_slice_count"] == 4
    assert result["slice_count_by_view"] == {"ambient": 2, "manifold": 2}
    assert result["landscape_recording_complete"] is True
    assert torch.isfinite(snapshots[-1][1]).all()
    terminal_slices = [
        entry for entry in result["slice_summaries"] if entry["phase"] == "terminal"
    ]
    assert {(entry["view"], entry["status"]) for entry in terminal_slices} == {
        ("ambient", "completed"),
        ("manifold", "invalid_center_off_manifold"),
    }
    assert result["finite_landscape_value_counts"]["ambient/mse"] == 50
    assert result["finite_landscape_value_counts"]["manifold/mse"] == 25
    assert fake_wandb.runs[0].finish_codes == [0]


def test_recording_failure_finishes_nonzero_without_completion_marker(monkeypatch):
    events = []
    fake_wandb = _install_mocks(monkeypatch, events)
    _instrument_training(monkeypatch)
    write_payload = recorded._write_payload

    def fail_trajectory_payload(artifact, name, payload):
        if name == "recorded_landscapes.npz":
            raise RuntimeError("forced in-memory artifact serialization failure")
        return write_payload(artifact, name, payload)

    monkeypatch.setattr(recorded, "_write_payload", fail_trajectory_payload)
    with pytest.raises(RuntimeError, match="forced in-memory artifact serialization failure"):
        recorded.main(_cpu_smoke_argv("oversized"))

    run = fake_wandb.runs[0]
    assert run.finish_codes == [1]
    assert run.summary["terminal_outcome"] == "infrastructure_failure"
    assert run.summary.get("landscape_recording_complete") is not True
    assert not any(event == ("recording_marker", True) for event in events)
