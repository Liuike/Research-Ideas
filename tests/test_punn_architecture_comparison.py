from copy import deepcopy

import pytest

from optimizer_resurrection.punn_architecture_comparison import build_comparison_report
from test_punn_architecture_analysis import adaptive_fixture_records, fixture_records


def _mark_failure(record, *, adaptive=False):
    result = record.summary["terminal_result"] if adaptive else record.summary
    result.update(terminal_outcome="numerical_failure", numerical_failure=True,
                  epochs_completed=0, failed_epoch=1,
                  failure_reason="nonfinite_gradient", failure_phase="backward",
                  initial_train_mse=None, train_mse=None, test_mse=None, objective=None)


def _comparison_fixtures():
    adaptive_config, adaptive_records = adaptive_fixture_records()
    baseline_config, baseline_records = fixture_records()
    for record in baseline_records:
        if record.config["task"] == "iris" and record.config["architecture"] == "small" and record.config["seed"] == 0:
            _mark_failure(record)
    for record in adaptive_records:
        if (record.config["task"] == "iris" and record.config["architecture"] == "small"
                and record.config["method"] == "muon_moonlight" and record.config["seed"] == 1):
            _mark_failure(record, adaptive=True)
    return adaptive_config, baseline_config, adaptive_records, baseline_records


def test_comparison_validates_cross_source_pairing_and_summarizes_only_failures():
    adaptive_config, baseline_config, adaptive_records, baseline_records = _comparison_fixtures()
    report = build_comparison_report(adaptive_config, baseline_config,
                                    adaptive_records, baseline_records)
    assert report["pairing_verified"] is True
    assert report["adaptive_expected_cells"] == report["adaptive_observed_cells"] == 12
    assert report["baseline_expected_cells"] == report["baseline_observed_cells"] == 6
    assert report["adaptive_source_revision"] == "adaptive-revision"
    assert report["baseline_source_revision"] == "revision"
    row = next(row for row in report["rows"]
               if row["task"] == "iris" and row["architecture"] == "small")
    assert row["sgd"]["numerical_failures"] == 1
    assert row["adamw"]["paired_with_sgd"]["sgd_only_failed"] == 1
    assert row["muon_moonlight"]["paired_with_sgd"]["adaptive_only_failed"] == 1
    assert "Moonlight Muon's internal" in report["numerical_failure_scope"]
    assert not any("mean_" in key for row in report["rows"] for key in row)


def test_comparison_rejects_adaptive_initialization_that_does_not_match_sgd():
    adaptive_config, baseline_config, adaptive_records, baseline_records = _comparison_fixtures()
    adaptive_records = deepcopy(adaptive_records)
    for record in adaptive_records:
        if (record.config["task"] == "iris" and record.config["architecture"] == "small"
                and record.config["seed"] == 0):
            record.summary["terminal_result"]["initialization_digest"] = "different-but-method-paired"
    with pytest.raises(ValueError, match="pairing mismatch"):
        build_comparison_report(adaptive_config, baseline_config,
                                adaptive_records, baseline_records)


def test_comparison_rejects_different_registered_seed_cells():
    adaptive_config, baseline_config, adaptive_records, baseline_records = _comparison_fixtures()
    baseline_config = deepcopy(baseline_config)
    baseline_config["seeds"] = [0]
    with pytest.raises(ValueError, match="differ in seeds"):
        build_comparison_report(adaptive_config, baseline_config,
                                adaptive_records, baseline_records)
