import copy
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from optimizer_resurrection import pure_cifar as pc
from optimizer_resurrection.pure_analysis import summarize_records, validate_landscape_history
from optimizer_resurrection.tracking import RunRecord


def config():
    return dict(protocol=pc.PROTOCOL, seeds=[0, 1, 2, 3, 4], data_seed_offset=100000,
                epochs=160, batch_size=128, learning_rate=.01, momentum=.9,
                weight_decay=.001, milestones=[80, 120], gamma=.1, device="cuda",
                num_workers=4, data_dir="data/cifar10", download=False,
                train_examples=50000, test_examples=10000, secondary=True,
                stage="reproduction", run_group=pc.PROTOCOL)


def smoke_args():
    return pc.parse_args(["--run-group", "test", "--stage", "test", "--device", "cpu",
                          "--epochs", "3", "--milestones", "1,2", "--num-workers", "0",
                          "--batch-size", "2", "--train-examples", "5", "--test-examples", "3"])


def loaders():
    gen = torch.Generator().manual_seed(42)
    train = TensorDataset(torch.randn(5, 2, generator=gen), torch.arange(5) % 2)
    test = TensorDataset(torch.randn(3, 2, generator=gen), torch.arange(3) % 2)
    return (DataLoader(train, batch_size=2), DataLoader(test, batch_size=2),
            dict(train_examples=5, test_examples=3, dataset_digest="fake", order_seed=1100003,
                 worker_seed=2100003, augmentation_seed=3100003))


def test_frozen_plan_and_recipe_rejection():
    commands = pc.expand_config(config())
    assert len(commands) == 5
    assert [pc.parse_args(c[3:]).seed for c in commands] == list(range(5))
    changed = config()
    changed["epochs"] = 159
    with pytest.raises(SystemExit):
        pc.expand_config(changed)


def test_plain_sgd_is_separate_and_both_recipes_are_frozen():
    plain = config()
    plain.update(protocol=pc.PLAIN_PROTOCOL, seeds=[0], momentum=0.0,
                 stage="exploratory", run_group=pc.PLAIN_PROTOCOL)
    commands = pc.expand_config(plain)
    assert len(commands) == 1
    args = pc.parse_args(commands[0][3:])
    assert args.seed == 0 and args.data_seed == 100000
    assert args.momentum == 0.0 and args.weight_decay == .001
    assert args.run_name == "pure-resnet18-cifar10-plain-sgd-seed0"
    for recipe, wrong in ((plain, .9), (config(), 0.0)):
        changed = {**recipe, "momentum": wrong}
        with pytest.raises(SystemExit):
            pc.expand_config(changed)
    changed = config()
    changed["secondary"] = False
    with pytest.raises(SystemExit):
        pc.expand_config(changed)


def test_schedule_coverage_final_evaluation_and_seed_replay(monkeypatch):
    records, calls = [], []
    original = pc.evaluate
    def spy(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(pc, "evaluate", spy)
    factory = lambda: torch.nn.Linear(2, 2)
    result = pc.run_trial(smoke_args(), records.append, loaders(), factory)
    assert result["terminal_outcome"] == "completed"
    assert result["epochs_completed"] == 3
    assert result["test_examples_evaluated"] == 3
    assert [r["train_examples_seen"] for r in records] == [5, 5, 5]
    assert [r["learning_rate"] for r in records] == pytest.approx([.01, .001, .0001])
    assert len(calls) == 1
    replay = pc.run_trial(smoke_args(), loaders=loaders(), model_factory=factory)
    for key in ("initialization_digest", "test_accuracy", "test_loss", "train_loss"):
        assert result[key] == replay[key]


def test_numerical_failure_skips_test_evaluation(monkeypatch):
    class Broken(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x) * float("nan")
    monkeypatch.setattr(pc, "evaluate", lambda *args: pytest.fail("test used after failure"))
    result = pc.run_trial(smoke_args(), loaders=loaders(), model_factory=lambda: Broken(2, 2))
    assert result["terminal_outcome"] == "numerical_failure"
    assert result["failure_reason"] == "nonfinite_loss"
    assert result["failed_epoch"] == 1 and result["epochs_completed"] == 0


def records():
    result = []
    for command in pc.expand_config(config()):
        args = pc.parse_args(command[3:])
        resolved = vars(args).copy()
        resolved["provenance"] = {"git": {"dirty": False, "revision": "clean"},
                                  "source_tree": {"digest": "source"}}
        summary = dict(terminal_outcome="completed", numerical_failure=False,
                       epochs_completed=160, test_examples_evaluated=10000,
                       train_examples=50000, test_examples=10000, test_accuracy=.9,
                       test_loss=.3, training_seconds=10., parameter_count=pc.PARAMETERS,
                       initialization_digest="init", dataset_digest="data",
                       order_seed=args.data_seed+1000003, worker_seed=args.data_seed+2000003,
                       augmentation_seed=args.data_seed+3000003)
        result.append(RunRecord(str(args.seed), "finished", resolved, summary, "url"))
    return result


def test_analysis_complete_and_duplicate_missing_dirty_rejection():
    data = records()
    report = summarize_records(config(), data)
    assert report["mean_test_accuracy_percent"] == 90
    assert report["sd_test_accuracy_percent"] == 0
    assert report["all_seeds_completed"]
    for bad in (data[:-1], data + [data[0]]):
        with pytest.raises(ValueError):
            summarize_records(config(), bad)
    bad = copy.deepcopy(data)
    bad[0].config["provenance"]["git"]["dirty"] = True
    with pytest.raises(ValueError):
        summarize_records(config(), bad)


def test_analysis_retains_explicit_failure():
    data = records()
    data[0] = replace(data[0], state="failed")
    data[0].summary.update(terminal_outcome="numerical_failure", numerical_failure=True,
                           failure_reason="nonfinite_gradient", failed_epoch=2, epochs_completed=1)
    report = summarize_records(config(), data)
    assert report["completed"] == 4 and report["numerical_failures"] == 1
    assert not report["all_seeds_completed"]


def test_analysis_plain_sgd_single_cell():
    plain = config()
    plain.update(protocol=pc.PLAIN_PROTOCOL, seeds=[0], momentum=0.0,
                 stage="exploratory", run_group=pc.PLAIN_PROTOCOL)
    args = pc.parse_args(pc.expand_config(plain)[0][3:])
    original = records()[0]
    resolved = {**vars(args), "provenance": original.config["provenance"]}
    record = replace(original, config=resolved)
    report = summarize_records(plain, [record])
    assert report["expected_cells"] == report["completed"] == 1
    assert report["sd_test_accuracy_percent"] is None


def test_landscape_protocol_preserves_legacy_condition_and_frozen_recipe():
    plain = config()
    plain.update(protocol=pc.PLAIN_PROTOCOL, seeds=[0], momentum=0.0,
                 stage="exploratory", run_group=pc.PLAIN_PROTOCOL)
    baseline = pc.parse_args(pc.expand_config(plain)[0][3:])
    assert not hasattr(baseline, "landscape")
    assert baseline.condition_id == "e0de80c7aa1b4e2420681baa9d18aebb6445dff80a175b8942cd152ebf83be54"
    landscape = {**plain, "protocol": pc.LANDSCAPE_PROTOCOL, "landscape": True,
                 "run_group": pc.LANDSCAPE_PROTOCOL}
    args = pc.parse_args(pc.expand_config(landscape)[0][3:])
    assert args.landscape and args.momentum == 0 and args.seed == 0
    assert args.condition_id != baseline.condition_id
    for key in ("epochs", "batch_size", "learning_rate", "momentum", "weight_decay",
                "milestones", "gamma", "seed", "data_seed"):
        assert getattr(args, key) == getattr(baseline, key)
    with pytest.raises(SystemExit):
        pc.expand_config({**landscape, "landscape": False})
    with pytest.raises(SystemExit):
        pc.expand_config({**landscape, "momentum": .9})


def test_landscape_training_replay_and_probe_schedule():
    baseline = smoke_args()
    baseline.momentum = 0.0
    baseline.protocol = pc.PLAIN_PROTOCOL
    models = []
    def factory():
        model = torch.nn.Linear(2, 2)
        models.append(model)
        return model
    reference = pc.run_trial(baseline, loaders=loaders(), model_factory=factory)
    instrumented = copy.deepcopy(baseline)
    instrumented.protocol = pc.LANDSCAPE_PROTOCOL
    instrumented.landscape = True
    recorded = []
    fixed = loaders()[1].dataset.tensors
    measured = pc.run_trial(instrumented, recorded.append, loaders(), factory, landscape_probe=fixed)
    for key in ("initialization_digest", "train_loss", "train_accuracy", "test_loss", "test_accuracy"):
        assert measured[key] == reference[key]
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key])
    assert measured["landscape_probe_epochs"] == [0, 1, 3]
    assert measured["landscape_gradient_batches"] == 9
    assert [row["epoch"] for row in recorded] == [0, 1, 2, 3]
    assert all(row["landscape/train/minibatch_count"] == 3 for row in recorded[1:])


def test_final_test_diagnostics_preserve_evaluation_and_detect_large_wrong_loss():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1000., 0.], [-1000., 0.]]))
        model.bias.zero_()
    loader = DataLoader(TensorDataset(torch.tensor([[1., 0.], [-1., 0.]]),
                                     torch.tensor([1, 1])), batch_size=2)
    regular = pc.evaluate(model, loader, torch.device("cpu"))
    measured = pc.evaluate(model, loader, torch.device("cpu"), diagnostics=True)
    assert all(measured[key] == value for key, value in regular.items())
    assert measured["landscape/test/loss_quantile_1"] == 2000
    assert measured["landscape/test/incorrect_mean_confidence"] == 1


def test_landscape_history_rejects_counts_without_measurements():
    cfg = {**config(), "protocol": pc.LANDSCAPE_PROTOCOL, "landscape": True,
           "seeds": [0], "momentum": 0.0, "run_group": pc.LANDSCAPE_PROTOCOL}
    fake = [{"epoch": epoch, "landscape/train/minibatch_count": 391,
             "landscape/train/gradient_valid_batches": 391,
             "landscape/train/nonfinite_gradient_batches": 0} for epoch in range(1, 161)]
    with pytest.raises(ValueError, match="gradient metric"):
        validate_landscape_history(cfg, {"epochs_completed": 160}, fake)


def test_analysis_rejects_running_or_impossible_failure():
    for state, completed, failed in (("running", 1, 2), ("failed", 999, 999),
                                      ("failed", 10, 2)):
        data = records()
        data[0] = replace(data[0], state=state)
        data[0].summary.update(terminal_outcome="numerical_failure", numerical_failure=True,
                              failure_reason="nonfinite_loss", failed_epoch=failed,
                              epochs_completed=completed)
        with pytest.raises(ValueError):
            summarize_records(config(), data)
