import sys
import types
from pathlib import Path

import pytest

import optimizer_resurrection.train as training


class FakeConfig(dict):
    def update(self, values=(), *, allow_val_change=False, **kwargs):
        assert allow_val_change
        return super().update(values, **kwargs)


class FakeRun:
    def __init__(self, kwargs, index):
        self.id = f"run-{index}"
        self.name = kwargs["name"]
        self.entity = kwargs["entity"]
        self.project = kwargs["project"]
        self.url = f"https://wandb.invalid/{self.id}"
        self.config = FakeConfig(kwargs["config"])
        self.summary = {}
        self.records = []
        self.finished = False
        self.exit_code = None

    def log(self, record, step=None):
        self.records.append((step, record))

    def finish(self, exit_code=None):
        self.finished = True
        self.exit_code = exit_code


class FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.runs = []
        self.init_calls = []

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        assert kwargs["mode"] == "online"
        assert kwargs["settings"]["mode"] == "online"
        assert kwargs["save_code"] is False
        assert "dir" not in kwargs
        self.init_calls.append(kwargs)
        run = FakeRun(kwargs, len(self.runs))
        self.runs.append(run)
        return run


@pytest.fixture
def fake_wandb(monkeypatch):
    module = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    return module


@pytest.mark.integration
def test_mlp_and_rnn_log_only_to_online_wandb(fake_wandb):
    project_root = Path(training.__file__).resolve().parents[2]
    before = {
        path.relative_to(project_root)
        for path in project_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }

    mlp = training.main([
        "--track", "mlp", "--dataset", "shapeset", "--activation", "logistic",
        "--depth", "3", "--width", "12", "--optimizer", "adam_o",
        "--steps", "2", "--batch-size", "4", "--train-examples", "16",
        "--validation-examples", "8", "--diagnostic-interval", "1",
        "--device", "cpu", "--stage", "test",
        "--run-name", "mlp-test", "--run-group", "tests",
    ])
    rnn = training.main([
        "--track", "rnn", "--task", "latch", "--hidden-size", "8",
        "--max-length", "10", "--optimizer", "adam_o",
        "--steps", "2", "--batch-size", "4", "--validation-examples", "8",
        "--diagnostic-interval", "1", "--device", "cpu", "--stage", "test",
        "--run-name", "rnn-test", "--run-group", "tests",
    ])

    after = {
        path.relative_to(project_root)
        for path in project_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert after == before
    assert mlp.train_loss == mlp.train_loss
    assert rnn.train_loss == rnn.train_loss
    assert len(fake_wandb.runs) == 2
    assert all(run.finished and run.exit_code == 0 for run in fake_wandb.runs)
    assert all(run.records for run in fake_wandb.runs)
    for run in fake_wandb.runs:
        assert run.summary["run_result"]["validation_accuracy"] >= 0.0
        assert "final_activation_saturation_mean" in run.summary
        assert "max_stiefel_residual" in run.summary
        assert len(run.config["condition_id"]) == 64
        provenance = run.config["provenance"]
        assert provenance["determinism"]["deterministic_algorithms"] is True
        assert provenance["determinism"]["deterministic_warn_only"] is False
        assert provenance["source_tree"]["file_count"] > 0
        assert provenance["wandb"]["id"] == run.id


def test_wandb_initialization_failure_happens_before_training(fake_wandb, monkeypatch):
    def fail_init(**kwargs):
        raise RuntimeError("cannot reach W&B")

    def training_must_not_start(*args, **kwargs):
        raise AssertionError("training started before W&B initialization")

    monkeypatch.setattr(fake_wandb, "init", fail_init)
    monkeypatch.setattr(training, "train_mlp", training_must_not_start)
    with pytest.raises(RuntimeError, match="cannot reach W&B"):
        training.main([
            "--track", "mlp", "--steps", "1", "--width", "4",
            "--depth", "2", "--batch-size", "2", "--device", "cpu",
            "--stage", "test",
        ])


def test_condition_id_ignores_run_identity_but_not_scientific_parameters():
    first = training.parse_args(["--track", "rnn", "--run-name", "attempt-a"])
    second = training.parse_args([
        "--track", "rnn", "--run-name", "attempt-b", "--run-group", "other",
    ])
    changed = training.parse_args([
        "--track", "rnn", "--run-name", "attempt-a", "--learning-rate", "0.2",
    ])
    assert training._condition_id(training._scientific_config(first)) == training._condition_id(
        training._scientific_config(second)
    )
    assert training._condition_id(training._scientific_config(first)) != training._condition_id(
        training._scientific_config(changed)
    )


def test_scientific_stages_require_clean_committed_source():
    with pytest.raises(RuntimeError, match="clean committed git revision"):
        training._require_reproducible_source(
            "calibration", {"revision": "abc", "dirty": True}
        )
    with pytest.raises(RuntimeError, match="clean committed git revision"):
        training._require_reproducible_source(
            "full-study", {"revision": None, "dirty": False}
        )
    training._require_reproducible_source(
        "engineering-smoke", {"revision": None, "dirty": True}
    )


def test_removed_local_logging_flags_are_rejected():
    with pytest.raises(SystemExit):
        training.parse_args(["--track", "mlp", "--output", "outputs/run"])
    with pytest.raises(SystemExit):
        training.parse_args(["--track", "mlp", "--wandb"])
