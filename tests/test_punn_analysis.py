from pathlib import Path

import pytest
import yaml

from optimizer_resurrection.punn_analysis import summarize_records
from optimizer_resurrection.punn_landscape import parse_args
from optimizer_resurrection.punn_protocol import expand_config
from optimizer_resurrection.tracking import RunRecord


CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs/product_unit/landscape_smoke.yaml")
    .read_text(encoding="utf-8")
)


def records() -> list[RunRecord]:
    result = []
    for index, command in enumerate(expand_config(CONFIG)):
        args = parse_args(command[3:])
        failed = args.method == "sgd"
        result.append(RunRecord(
            str(index), "failed" if failed else "finished", vars(args),
            {"terminal_outcome": "numerical_failure" if failed else "completed",
             "numerical_failure": failed, "epochs_completed": 0 if failed else args.epochs,
             "initial_train_mse": 1.0, "train_mse": None if failed else 0.1,
             "test_mse": None if failed else 0.2,
             "dataset_digest": f"{args.task}-{args.seed}"},
            f"https://example.invalid/{index}",
        ))
    return result


def test_analysis_accepts_explicit_failures_and_requires_all_cells() -> None:
    report = summarize_records(CONFIG, records())
    assert report["expected_cells"] == 6
    assert sum(row["numerical_failures"] for row in report["rows"]) == 2
    assert all(row["sgd_failed_or_higher_test_mse_than_both"] == 1
               for row in report["paired_rows"])
    with pytest.raises(ValueError, match="incomplete PUNN plan"):
        summarize_records(CONFIG, records()[:-1])


def test_analysis_rejects_duplicates_and_broken_pairs() -> None:
    runs = records()
    with pytest.raises(ValueError, match="duplicate PUNN condition"):
        summarize_records(CONFIG, [*runs, runs[0]])
    bad = list(runs)
    bad[1] = RunRecord(bad[1].run_id, bad[1].state, bad[1].config,
                       {**bad[1].summary, "dataset_digest": "different"}, bad[1].url)
    with pytest.raises(ValueError, match="different datasets"):
        summarize_records(CONFIG, bad)
