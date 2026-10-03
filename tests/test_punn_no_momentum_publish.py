from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from optimizer_resurrection import punn_no_momentum_publish as publisher
from optimizer_resurrection.punn_architecture_stability import MANIFOLD_METHOD, MANIFOLD_RECIPE
from optimizer_resurrection.tracking import RunRecord


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "configs/product_unit/manifold_architecture_stability.yaml"
HISTORICAL_REVISION = "historical-clean-revision"
HISTORICAL_DIGEST = "historical-source-digest"


@pytest.fixture(scope="module")
def historical_anchor_fixture():
    config = yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))
    historical = {
        "protocol": "punn-manifold-architecture-comparison-v1",
        "pairing_verified": True,
        "manifold_source_revision": HISTORICAL_REVISION,
        "manifold_recorded_source_tree_digests": [HISTORICAL_DIGEST],
        "manifold_recipe": dict(MANIFOLD_RECIPE),
        "manifold_source_run_ids": [],
        "excluded_infrastructure_runs": [{"run_id": "declared-infra-attempt"}],
        "rows": [],
    }
    legacy_records = []
    new_rows = []
    for cell_index, condition in enumerate(config["conditions"]):
        task, architecture = condition["task"], condition["architecture"]
        completed = []
        for seed in config["seeds"]:
            run_id = f"historical-{task}-{architecture}-{seed}"
            data_seed = seed + config["data_seed_offset"]
            dataset_digest = f"dataset-{cell_index}-{seed}"
            initialization_digest = f"preprojection-{cell_index}-{seed}"
            order_seed = seed + 1_000_003
            train_mse = (cell_index * 30 + seed + 1) / 1000
            test_mse = train_mse + 0.1
            result = {
                "terminal_outcome": "completed",
                "epochs_completed": 500,
                "data_seed": data_seed,
                "dataset_digest": dataset_digest,
                "preprojection_initialization_digest": initialization_digest,
                "order_seed": order_seed,
                "train_mse": train_mse,
                "test_mse": test_mse,
            }
            record_config = {
                "task": task,
                "architecture": architecture,
                "method": MANIFOLD_METHOD,
                "seed": seed,
                "data_seed": data_seed,
                **MANIFOLD_RECIPE,
                "provenance": {
                    "git": {"revision": HISTORICAL_REVISION, "dirty": False},
                    "source_tree": {"digest": HISTORICAL_DIGEST},
                },
            }
            legacy_records.append(RunRecord(
                run_id, "finished", record_config, {"terminal_result": result},
                f"https://wandb.test/{run_id}",
            ))
            historical["manifold_source_run_ids"].append(run_id)
            completed.append({
                "run_id": run_id,
                "seed": seed,
                "train_mse": train_mse,
                "test_mse": test_mse,
            })
            new_rows.append({
                "condition": {
                    "task": task,
                    "architecture": architecture,
                    "seed": seed,
                    "data_seed": data_seed,
                },
                "dataset_digest": dataset_digest,
                "preprojection_initialization_digest": initialization_digest,
                "order_seed": order_seed,
            })
        historical["rows"].append({
            "task": task,
            "architecture": architecture,
            "manifold_muon_da10": {
                "final_performance": {"completed": completed},
            },
        })

    assert len(legacy_records) == len(new_rows) == 480
    assert len(historical["manifold_source_run_ids"]) == 480
    return historical, legacy_records, new_rows


def test_full_historical_anchor_pairs_all_cells_and_preserves_declared_infrastructure_attempt(
    historical_anchor_fixture,
):
    historical, legacy_records, new_rows = deepcopy(historical_anchor_fixture)
    legacy_records.append(RunRecord(
        "declared-infra-attempt", "failed", {},
        {"terminal_result": {"terminal_outcome": "infrastructure_failure"}},
        "https://wandb.test/declared-infra-attempt",
    ))

    result = publisher.verify_historical_anchor(historical, legacy_records, new_rows)

    assert result["verified_cells"] == 480
    assert result["dataset_digest_exact"] is True
    assert result["preprojection_initialization_exact"] is True
    assert result["order_seed_exact"] is True


def test_historical_anchor_rejects_changed_data_initialization_order_recipe_or_source(
    historical_anchor_fixture,
):
    original = historical_anchor_fixture
    first_record_id = original[1][0].run_id
    cases = (
        "data_seed",
        "dataset_digest",
        "preprojection_initialization_digest",
        "order_seed",
        "recipe",
        "source_revision",
        "source_digest",
    )
    for case in cases:
        historical, legacy_records, new_rows = deepcopy(original)
        record = next(item for item in legacy_records if item.run_id == first_record_id)
        if case == "data_seed":
            record.config["data_seed"] += 1
        elif case in {"dataset_digest", "preprojection_initialization_digest", "order_seed"}:
            record.summary["terminal_result"][case] = "changed"
        elif case == "recipe":
            record.config["momentum"] = 0.9
        elif case == "source_revision":
            record.config["provenance"]["git"]["revision"] = "other-revision"
        else:
            record.config["provenance"]["source_tree"]["digest"] = "other-digest"

        with pytest.raises(ValueError, match="historical anchor|historical/new"):
            publisher.verify_historical_anchor(historical, legacy_records, new_rows)


def test_historical_anchor_rejects_unregistered_attempt_even_when_other_480_match(
    historical_anchor_fixture,
):
    historical, legacy_records, new_rows = deepcopy(historical_anchor_fixture)
    legacy_records.append(RunRecord(
        "undeclared-attempt", "failed", {}, {}, "https://wandb.test/undeclared-attempt",
    ))

    with pytest.raises(ValueError, match="unregistered historical attempts"):
        publisher.verify_historical_anchor(historical, legacy_records, new_rows)


def test_historical_anchor_rejects_historical_sample_from_another_run(
    historical_anchor_fixture,
):
    historical, legacy_records, new_rows = deepcopy(historical_anchor_fixture)
    historical["rows"][0]["manifold_muon_da10"]["final_performance"]["completed"][0]["run_id"] = (
        "different-run"
    )

    with pytest.raises(ValueError, match="historical sample identity mismatch"):
        publisher.verify_historical_anchor(historical, legacy_records, new_rows)
