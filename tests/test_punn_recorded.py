import io
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from optimizer_resurrection.experiment import expand_config
from optimizer_resurrection.punn_landscape import make_dataset, run_trial
from optimizer_resurrection.punn_recorded import PROTOCOL, LandscapeRecorder, _add_cpu_reference, _write_payload, parse_args
from optimizer_resurrection.tracking import RunRecord
from optimizer_resurrection.punn_analysis import summarize_records

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_plan_preserves_recipe_and_has_separate_conditions():
    config = yaml.safe_load((ROOT / "configs/product_unit/landscape_recorded.yaml").read_text())
    commands = expand_config(config)
    assert len(commands) == 180
    parsed = [parse_args(c[3:]) for c in commands]
    assert len({a.condition_id for a in parsed}) == 180
    for args in parsed:
        assert args.epochs == 500 and args.population_size == 20
        assert args.learning_rate == 0.1 and args.momentum == 0.9
        assert args.data_seed == args.seed + 100000
        assert args.device == "cpu" and args.landscape_device == "cuda"
        assert args.protocol == PROTOCOL
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--slice-points", "4"])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--condition-id", "incorrect"])
    reference_config = yaml.safe_load((ROOT / "configs/product_unit/landscape_recorded_references.yaml").read_text())
    references = [parse_args(c[3:]) for c in expand_config(reference_config)]
    assert len(references) == 180
    assert all(a.landscape_cpu_reference == "true" for a in references)
    assert {a.condition_id for a in parsed}.isdisjoint({a.condition_id for a in references})


def test_cpu_reference_uses_original_model_forward_and_preserves_raw_disagreements():
    from optimizer_resurrection.models import ProductUnitNetwork
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    from types import SimpleNamespace

    model = ProductUnitNetwork(2, 3)
    dataset = make_dataset("f4", 100000)
    vector = torch.tensor([-7., -7., 7., 7., -7., -7., 7., 7., -7., -7.])
    reference_model = ProductUnitNetwork(2, 3)
    vector_to_parameters(vector, reference_model.parameters())
    with torch.no_grad():
        expected = (reference_model(dataset.train_x)-dataset.train_y).square().mean()
    before_model = parameters_to_vector(model.parameters()).detach().clone()
    before_rng = torch.get_rng_state().clone()
    record = {"vectors": vector.repeat(16, 1), "mse": torch.full((16,), float("inf"))}
    _add_cpu_reference(record, model, dataset, SimpleNamespace(landscape_device="cuda",landscape_batch_size=16))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert torch.equal(parameters_to_vector(model.parameters()), before_model)
    assert torch.equal(record["cpu_reference_mse"], expected.repeat(16))
    assert torch.isfinite(expected)
    assert torch.isinf(record["mse"]).all()
    assert record["precision_comparison"]["finite_mask_disagreements"] == 16


def test_original_recorded_condition_ids_remain_compatible_without_reference():
    config = yaml.safe_load((ROOT / "configs/product_unit/landscape_recorded_cpu_smoke.yaml").read_text())
    assert parse_args(expand_config(config)[0][3:]).condition_id == "544ae0a928286c72306a4aefeca3ace31caa477ded9f6c474adacc4f531c0b6f"


def test_adamw_recorded_plan_keeps_existing_recipe_and_requires_secondary_decay():
    config = yaml.safe_load((ROOT / "configs/product_unit/adamw_landscape.yaml").read_text())
    commands = expand_config(config)
    assert len(commands) == 20
    parsed = [parse_args(command[3:]) for command in commands]
    assert len({args.condition_id for args in parsed}) == 20
    for args in parsed:
        assert args.method == "adamw" and args.protocol == "punn-adamw-recorded-v1"
        assert args.learning_rate == .001 and args.weight_decay == .01
        assert args.secondary == "true" and args.epochs == 500
        assert args.data_seed == 100000 + args.seed
        assert args.landscape_cpu_reference == "true"
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--secondary", "false"])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--weight-decay", "nan"])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--protocol", PROTOCOL])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--method", "sgd"])


def test_muon_recorded_plan_matches_frozen_gradient_recipe_and_validates_head_rate():
    config = yaml.safe_load((ROOT / "configs/product_unit/muon_landscape.yaml").read_text())
    recipe = yaml.safe_load((ROOT / "configs/product_unit/gradient_defaults.yaml").read_text())["recipes"]["muon_moonlight"]
    commands = expand_config(config)
    assert len(commands) == 20
    parsed = [parse_args(command[3:]) for command in commands]
    assert len({args.condition_id for args in parsed}) == 20
    for args in parsed:
        assert args.method == "muon_moonlight" and args.protocol == "punn-muon-recorded-v1"
        for key in ("learning_rate", "momentum", "weight_decay", "aux_learning_rate"):
            assert getattr(args, key) == recipe[key]
        assert args.secondary == "false" and args.epochs == 500
        assert args.data_seed == 100000 + args.seed
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--aux-learning-rate", "0"])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--aux-learning-rate", "nan"])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--method", "adamw"])
    with pytest.raises(SystemExit):
        parse_args([*commands[0][3:], "--weight-decay", "0.01"])


@pytest.mark.parametrize("method", ["sgd", "pso", "de"])
def test_state_recording_preserves_exact_training_result_and_keeps_terminal(method):
    from optimizer_resurrection.models import ProductUnitNetwork
    from optimizer_resurrection.punn_sampling import orthonormal_directions, sample_slice
    kwargs = dict(task="f1", method=method, seed=3, data_seed=100003,
                  epochs=2, population_size=4)
    expected = run_trial(**kwargs)
    dataset = make_dataset("f1", 100003)
    observer_model = ProductUnitNetwork(1, 1)
    directions = orthonormal_directions(3, 97)
    states = []

    def observer(epoch, kind, vector, population, losses):
        states.append((epoch, kind, vector.clone()))
        if torch.isfinite(vector).all():
            sampled = sample_slice(observer_model, dataset.train_x, dataset.train_y, vector,
                                   directions=directions, points=3, device="cpu")
            from types import SimpleNamespace
            _add_cpu_reference(sampled, observer_model, dataset,
                               SimpleNamespace(landscape_device="cpu", landscape_batch_size=32))
        # Mutating the callback copies must not mutate the training model.
        vector.fill_(100)
        if population is not None:
            population.fill_(100)
            losses.fill_(100)

    actual = run_trial(**kwargs, on_state=observer)
    assert asdict(actual) == asdict(expected)
    assert states[0][1] == "initial"
    assert states[-1][1] == actual.terminal_outcome.replace("completed", "final")


def test_artifact_arrays_load_without_pickle_and_preserve_nonfinite_samples():
    class MemoryArtifact:
        @contextmanager
        def new_file(self, name, mode):
            buffer = io.BytesIO()
            yield buffer
            self.value = buffer.getvalue()

    artifact = MemoryArtifact()
    _write_payload(artifact, "samples.npz", {"sample": {"mse": torch.tensor([1.0, float("inf")]),
                                                      "nonfinite": torch.tensor([False, True])},
                                            "note": "unclamped"})
    with np.load(io.BytesIO(artifact.value), allow_pickle=False) as archive:
        assert np.isinf(archive["sample/mse"][1])
        assert archive["sample/nonfinite"].tolist() == [False, True]
        assert json.loads(str(archive["metadata_json"]))["note"] == "unclamped"


@pytest.mark.parametrize("task", ["f1", "f4"])
def test_recorder_table_plane_coordinates_and_saved_states(task):
    from optimizer_resurrection.models import ProductUnitNetwork
    from torch.nn.utils import parameters_to_vector
    from optimizer_resurrection.punn_landscape import TASKS

    class MemoryArtifact:
        def __init__(self):
            self.files = {}
        @contextmanager
        def new_file(self, name, mode):
            buffer = io.BytesIO()
            yield buffer
            self.files[name] = buffer.getvalue()
        def add(self, value, name):
            self.table = value

    args = parse_args(["--task", task, "--method", "sgd", "--seed", "0",
                       "--data-seed", "100000", "--run-group", "test", "--slice-points", "5",
                       "--landscape-device", "cpu"])
    model = ProductUnitNetwork(*TASKS[task][:2])
    artifact = MemoryArtifact()
    recorder = LandscapeRecorder(args, model, make_dataset(task, args.data_seed), None, artifact)
    vector = parameters_to_vector(model.parameters()).detach().clone()
    recorder(0, "initial", vector, None, None)
    recorder(1, "epoch", vector, None, None)
    recorder(1, "final", vector, None, None)
    counts = recorder.finish()
    assert counts["landscape_states"] == 3
    assert counts["landscape_slices"] == 2
    assert len(artifact.table.data) == 50
    assert [(row[2], row[3]) for row in artifact.table.data[:25]] == [
        (u, v) for u in [-1., -.5, 0., .5, 1.] for v in [-1., -.5, 0., .5, 1.]]
    assert all(row[-2:] == [None, None] for row in artifact.table.data)
    with np.load(io.BytesIO(artifact.files["training_landscapes.npz"]), allow_pickle=False) as archive:
        assert archive["slices/0000/sample/grid_coordinates"].shape == (5, 5, vector.numel())
        assert np.array_equal(archive["states/0000/parameters"], vector.numpy())


def test_recorded_analysis_requires_landscapes_and_initialization_pairing():
    config = yaml.safe_load((ROOT / "configs/product_unit/landscape_recorded_cpu_reference_smoke.yaml").read_text())
    records = []
    for index, command in enumerate(expand_config(config)):
        args = parse_args(command[3:])
        summary = {"terminal_outcome": "completed", "numerical_failure": False,
                   "epochs_completed": args.epochs, "train_mse": 0.1, "test_mse": 0.1,
                   "initial_train_mse": 1.0, "dataset_digest": args.task,
                   "initialization_digest": args.task, "order_seed": args.seed + 1_000_003,
                   "landscape_status": "completed", "landscape_artifact": f"artifact-{index}:v0",
                   "landscape_slices": 2, "landscape_seed": args.data_seed + args.landscape_seed_offset,
                   "landscape_samples_digest": args.task, "landscape_cpu_reference": True}
        records.append(RunRecord(str(index), "finished", vars(args), summary, "https://example.invalid"))
    assert summarize_records(config, records)["observed_cells"] == 6
    first = records[0]
    bad = RunRecord(first.run_id, first.state, first.config,
                    {**first.summary, "landscape_status": "failed"}, first.url)
    with pytest.raises(ValueError, match="landscape artifact"):
        summarize_records(config, [bad, *records[1:]])
    bad = RunRecord(first.run_id, first.state, first.config,
                    {**first.summary, "initialization_digest": "other"}, first.url)
    with pytest.raises(ValueError, match="different initializations"):
        summarize_records(config, [bad, *records[1:]])
    bad = RunRecord(first.run_id, first.state, first.config,
                    {**first.summary, "landscape_cpu_reference": False}, first.url)
    with pytest.raises(ValueError, match="CPU landscape reference"):
        summarize_records(config, [bad, *records[1:]])


def test_concurrent_plan_attempts_all_registered_commands_and_reports_execution_failure(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from optimizer_resurrection import experiment

    config = tmp_path / "plan.yaml"
    config.write_text("protocol: test\n")
    commands = [["python", "-m", "test.module", "--seed", str(i)] for i in range(4)]
    monkeypatch.setattr(experiment, "expand_config", lambda config, recipes: commands)
    seen = []
    lock = threading.Lock()

    def execute(command, check):
        assert check is False
        with lock:
            seen.append(command)
        return SimpleNamespace(returncode=1 if command[-1] == "0" else 0)

    monkeypatch.setattr(experiment.subprocess, "run", execute)
    monkeypatch.setattr(experiment.sys, "argv", ["experiment", "plan", str(config), "--run", "--workers", "2"])
    with pytest.raises(SystemExit) as failure:
        experiment.main()
    assert failure.value.code == 1
    assert sorted(c[-1] for c in seen) == ["0", "1", "2", "3"]
    assert all(c[0] == experiment.sys.executable for c in seen)
