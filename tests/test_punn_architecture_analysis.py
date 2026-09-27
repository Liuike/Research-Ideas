from copy import deepcopy

import pytest

from optimizer_resurrection.punn_architecture_analysis import summarize_records
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
