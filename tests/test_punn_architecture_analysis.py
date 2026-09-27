from copy import deepcopy
from pathlib import Path

import pytest

from optimizer_resurrection.punn_architecture_analysis import (
    _terminal_snapshot,
    summarize_adaptive_records,
    summarize_records,
)
from optimizer_resurrection.tracking import RunRecord


def fixture_records():
    from optimizer_resurrection.punn_architecture_stability import expand_config, parse_args
    from optimizer_resurrection.punn_architecture_data import RAW_SOURCES
    config = {"protocol": "punn-architecture-stability-v1",
              "conditions": [{"task": "iris", "architecture": a}
                             for a in ["small", "oversized", "regularized"]],
              "seeds": [0, 1], "data_seed_offset": 100000, "epochs": 2,
              "learning_rate": 0.1, "momentum": 0.0, "regularization_lambda": 0.0001,
              "log_every": 1, "device": "cpu", "stage": "exploratory", "run_group": "test"}
    records = []
    for i, command in enumerate(expand_config(config)):
        args = parse_args(command[3:])
        resolved = vars(args).copy()
        resolved["provenance"] = {"git": {"dirty": False, "revision": "revision"},
                                  "source_tree": {"digest": "source"}}
        summary = {"terminal_outcome": "completed", "numerical_failure": False,
                   "epochs_completed": 2, "initial_train_mse": 1.0,
                   "train_mse": .2, "test_mse": .3, "objective": .21,
                   "dataset_digest": str(args.seed), "order_seed": args.seed + 1_000_003,
                   "initialization_digest": str(args.seed) + ("small" if args.architecture == "small" else "large")}
        artifact = "dataset-" + str(args.seed) + ":v0"
        source = RAW_SOURCES[args.task]
        summary["dataset_artifact"] = resolved["dataset_artifact"] = artifact
        resolved["data_metadata"] = {"task": args.task, "data_seed": args.data_seed,
                                     "dataset_digest": summary["dataset_digest"],
                                     "raw_source": {"url": source.url, "sha256": source.sha256}}
        records.append(RunRecord(str(i), "finished", resolved, summary, "url"))
    return config, records


def test_accepts_complete_pairs_and_counts_failure_separately():
    config, records = fixture_records()
    summary = records[0].summary
    summary.update(terminal_outcome="numerical_failure", numerical_failure=True,
                   epochs_completed=0, failed_epoch=1, failure_reason="nonfinite_gradient",
                   failure_phase="backward", train_mse=None, test_mse=None)
    report = summarize_records(config, records)
    assert report["expected_cells"] == report["observed_cells"] == 6
    row = next(row for row in report["rows"] if row["architecture"] == "small")
    assert row["numerical_failures"] == 1
    assert row["mean_test_mse"] == .3


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "data", "initialization", "dirty", "source", "nonfinite", "incomplete", "artifact", "raw_source", "failed_state"])
def test_rejects_unreviewable_records(mutation):
    config, original = fixture_records()
    records = deepcopy(original)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "data":
        records[-1].summary["dataset_digest"] = "different"
    elif mutation == "initialization":
        records[-1].summary["initialization_digest"] = "different"
    elif mutation == "dirty":
        records[0].config["provenance"]["git"]["dirty"] = True
    elif mutation == "source":
        records[0].config["provenance"]["source_tree"]["digest"] = "other"
    elif mutation == "nonfinite":
        records[0].summary["test_mse"] = float("inf")
    elif mutation == "incomplete":
        records[0].summary["epochs_completed"] = 1
    elif mutation == "artifact":
        records[0].summary.pop("dataset_artifact")
    elif mutation == "raw_source":
        records[0].config["data_metadata"]["raw_source"]["sha256"] = "wrong"
    elif mutation == "failed_state":
        r = records[0]
        r.summary.update(terminal_outcome="numerical_failure", numerical_failure=True,
                         failed_epoch=1, failure_reason="nonfinite_gradient", failure_phase="backward")
        records[0] = RunRecord(r.run_id, "failed", r.config, r.summary, r.url)
    with pytest.raises(ValueError):
        summarize_records(config, records)


def adaptive_fixture_records():
    import yaml
    from optimizer_resurrection.punn_architecture_stability import expand_config, parse_args
    from optimizer_resurrection.punn_architecture_data import RAW_SOURCES

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/product_unit/adaptive_architecture_stability.yaml").read_text())
    config.update({"conditions": [{"task": "iris", "architecture": architecture}
                                  for architecture in ["small", "oversized", "regularized"]],
                   "seeds": [0, 1], "epochs": 2, "log_every": 1,
                   "stage": "exploratory", "run_group": "adaptive-test"})
    records = []
    for index, command in enumerate(expand_config(config)):
        args = parse_args(command[3:])
        resolved = vars(args).copy()
        resolved["provenance"] = {"git": {"dirty": False, "revision": "adaptive-revision"},
                                  "source_tree": {"digest": "adaptive-source"}}
        data_digest = str(args.seed)
        large = args.architecture in {"oversized", "regularized"}
        result = {
            "condition_id": args.condition_id, "protocol": args.protocol,
            "task": args.task, "architecture": args.architecture, "method": args.method,
            "seed": args.seed, "data_seed": args.data_seed, "epochs": args.epochs,
            "input_dim": args.input_dim, "hidden_units": args.hidden_units,
            "output_dim": args.output_dim, "regularization_lambda": args.regularization_lambda,
            "learning_rate": args.learning_rate, "momentum": args.momentum,
            "weight_decay": args.weight_decay, "aux_learning_rate": args.aux_learning_rate,
            "secondary": args.secondary, "terminal_outcome": "completed",
            "numerical_failure": False, "epochs_completed": 2,
            "initial_train_mse": 1.0, "train_mse": .2, "test_mse": .3, "objective": .21,
            "failed_epoch": None, "failure_phase": None, "failure_reason": None,
            "dataset_digest": data_digest,
            "initialization_digest": f"{args.seed}{'large' if large else 'small'}",
            "order_seed": args.seed + 1_000_003,
            "dataset_artifact": f"dataset-iris-{args.data_seed}:v0",
        }
        source = RAW_SOURCES[args.task]
        resolved["data_metadata"] = {"task": args.task, "data_seed": args.data_seed,
                                      "dataset_digest": data_digest,
                                      "raw_source": {"url": source.url, "sha256": source.sha256}}
        resolved["dataset_artifact"] = result["dataset_artifact"]
        records.append(RunRecord(str(index), "finished", resolved,
                                 {"terminal_result": result}, "url"))
    return config, records


def test_adaptive_summary_groups_methods_and_accepts_flattened_wandb_snapshot():
    config, records = adaptive_fixture_records()
    record = next(record for record in records
                  if record.config["method"] == "muon_moonlight")
    result = record.summary.pop("terminal_result")
    result.update(terminal_outcome="numerical_failure", numerical_failure=True,
                  epochs_completed=0, failed_epoch=1, failure_phase="training_forward",
                  failure_reason="nonfinite_prediction", initial_train_mse=None,
                  train_mse=None, test_mse=None, objective=None)
    result["failure_pattern"] = {
        "tensor_names": ["exponents"],
        "nonfinite_values": 2,
        "examples": [{"tensor": "exponents", "index": [0, 1]}],
    }
    result["optimizer_betas"] = {"auxiliary_adamw": [0.9, 0.95]}
    for key, value in result.items():
        if key in {"failure_pattern", "optimizer_betas"}:
            record.summary.update({f"terminal_result.{key}.{child}": child_value
                                   for child, child_value in value.items()})
        else:
            record.summary["terminal_result." + key] = value
    report = summarize_adaptive_records(config, records)
    snapshot = _terminal_snapshot(record.summary, record.run_id)
    assert report["expected_cells"] == report["observed_cells"] == 12
    assert {row["method"] for row in report["rows"]} == {"adamw", "muon_moonlight"}
    assert len(report["rows"]) == 6
    assert all(row["attempted"] == 2 for row in report["rows"])
    assert report["source_revision"] == "adaptive-revision"
    assert "Moonlight Muon" in report["numerical_failure_scope"]
    assert snapshot["failure_pattern"] == result["failure_pattern"]
    assert snapshot["optimizer_betas"] == result["optimizer_betas"]
    assert next(row for row in report["rows"]
                if row["task"] == record.config["task"]
                and row["architecture"] == record.config["architecture"]
                and row["method"] == "muon_moonlight")["numerical_failures"] == 1


def test_terminal_snapshot_merges_mixed_paths_without_mutating_summary():
    parent = {"tensor_names": ["exponents"], "metadata": {"source_values": [1]}}
    dotted_parent = {"dotted_values": [2]}
    summary = {
        # Descendants deliberately precede their parents in insertion order.
        "terminal_result.failure_pattern.metadata.marker": True,
        "terminal_result.failure_pattern.metadata": dotted_parent,
        "terminal_result.failure_pattern": parent,
    }
    original = deepcopy(summary)

    snapshot = _terminal_snapshot(summary, "mixed-encoding")

    assert snapshot["failure_pattern"] == {
        "tensor_names": ["exponents"],
        "metadata": {"source_values": [1], "dotted_values": [2], "marker": True},
    }
    assert summary == original
    snapshot["failure_pattern"]["metadata"]["source_values"].append(3)
    snapshot["failure_pattern"]["metadata"]["dotted_values"].append(4)
    assert summary == original


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "identity", "pairing", "no_snapshot"])
def test_adaptive_summary_rejects_incomplete_or_mismatched_records(mutation):
    from copy import deepcopy

    config, original = adaptive_fixture_records()
    records = deepcopy(original)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "identity":
        records[0].summary["terminal_result"]["seed"] = 29
    elif mutation == "pairing":
        records[0].summary["terminal_result"]["initialization_digest"] = "mismatched"
    elif mutation == "no_snapshot":
        records[0].summary.clear()
    with pytest.raises(ValueError):
        summarize_adaptive_records(config, records)
