"""Publish the complete, paired DA-10 no-momentum graph bundle from W&B."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import torch
import yaml

from .punn_architecture_analysis import _terminal_snapshot
from .punn_manifold_landscape_report import collect_completed, download_entry, records, same_snapshot
from .punn_no_momentum_check import check_group
from .tracking import load_wandb_credentials, require_online_wandb, wandb_analysis_run
from .train import _git_metadata, _require_reproducible_source, _source_tree_digest

REVISION = "fb8f9b1afaaf9a0dbdc7e003d55892edb626a876"
DIGEST = "15123dbfdb7ae923f6bbfe6e946c9cf57aaa502364fdee4ae3a5c737db76b65f"
REFERENCE = "punn-da10-recorded-landscape-comparison-e00u5oe9:v0"
PERFORMANCE_REFERENCE = "punn-final-performance-4761ik87:v0"


def verify_reference_cache(accepted, reference, project, cache, workers):
    """Bind the prior audit's rows to exact live immutable raw artifact entries."""
    import wandb
    indexed = {row["run_id"]: row for row in reference["runs"]}
    if len(indexed) != 480 or set(indexed) != {r.run_id for r in accepted}:
        raise ValueError("reference report differs from strictly reconciled accepted outcomes")
    local = threading.local()

    def verify(record):
        if not hasattr(local, "api"):
            local.api = wandb.Api(timeout=120)
        row = indexed[record.run_id]
        result = _terminal_snapshot(record.summary, record.run_id)
        if (row["terminal_outcome"] != result["terminal_outcome"]
                or row["source_revision"] != record.config["provenance"]["git"]["revision"]
                or row["source_digest"] != record.config["provenance"]["source_tree"]["digest"]):
            raise ValueError(f"reference outcome/source mismatch: {record.run_id}")
        for metric in ("train_mse", "test_mse"):
            if not same_snapshot(row.get(metric), result.get(metric)):
                raise ValueError(f"reference terminal metric mismatch: {record.run_id}: {metric}")
        artifact = local.api.artifact(result["landscape_artifact"], type="loss-landscape")
        if artifact.qualified_name != row["artifact"]:
            raise ValueError(f"reference artifact identity mismatch: {record.run_id}")
        directory = cache / "runs" / record.run_id
        for name in ("manifest.json", "provenance.json", "recorded_landscapes.npz"):
            download_entry(artifact, name, directory)
        dataset = local.api.artifact(row["dataset_artifact"], type="dataset")
        download_entry(dataset, "dataset.npz", directory / "dataset")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify, accepted))


def verify_historical_anchor(historical, legacy_records, new_rows):
    """Pair the prior four-way audit to new data/init/order through its DA-10 arm."""
    if (historical.get("protocol") != "punn-manifold-architecture-comparison-v1"
            or historical.get("pairing_verified") is not True):
        raise ValueError("historical optimizer report lacks its certified pairing contract")
    new = {(row["condition"]["task"], row["condition"]["architecture"], row["condition"]["seed"]): row
           for row in new_rows}
    samples = {(cell["task"], cell["architecture"], sample["seed"]): sample
               for cell in historical["rows"]
               for sample in cell["manifold_muon_da10"]["final_performance"]["completed"]}
    registered_ids = set(historical["manifold_source_run_ids"])
    excluded_ids = {r["run_id"] for r in historical.get("excluded_infrastructure_runs", [])}
    unknown = {r.run_id for r in legacy_records} - registered_ids - excluded_ids
    if unknown:
        raise ValueError(f"unregistered historical attempts: {sorted(unknown)}")
    legacy_records = [r for r in legacy_records if r.run_id in registered_ids]
    if len(new) != 480 or len(samples) != 480 or len(legacy_records) != 480:
        raise ValueError("historical anchor requires every registered 30-seed cell")
    allowed_digests = set(historical["manifold_recorded_source_tree_digests"])
    seen = set()
    for record in legacy_records:
        config = record.config
        key = (config.get("task"), config.get("architecture"), config.get("seed"))
        if key not in new or key in seen or record.state != "finished":
            raise ValueError(f"unknown/duplicate/incomplete historical anchor: {record.run_id}")
        seen.add(key)
        result = _terminal_snapshot(record.summary, record.run_id)
        row = new[key]
        source = config["provenance"]
        if (source["git"].get("dirty") is not False
                or source["git"].get("revision") != historical["manifold_source_revision"]
                or source["source_tree"].get("digest") not in allowed_digests):
            raise ValueError(f"historical anchor source mismatch: {record.run_id}")
        for field, value in historical["manifold_recipe"].items():
            if config.get(field) != value:
                raise ValueError(f"historical anchor recipe mismatch: {record.run_id}: {field}")
        if (result.get("terminal_outcome") != "completed" or result.get("epochs_completed") != 500
                or config.get("data_seed") != row["condition"]["data_seed"]
                or result.get("dataset_digest") != row["dataset_digest"]
                or result.get("preprojection_initialization_digest") != row["preprojection_initialization_digest"]
                or result.get("order_seed") != row["order_seed"]):
            raise ValueError(f"historical/new data or initialization mismatch: {record.run_id}")
        sample = samples[key]
        if sample["run_id"] != record.run_id:
            raise ValueError(f"historical sample identity mismatch: {record.run_id}")
        for metric in ("train_mse", "test_mse"):
            if not same_snapshot(sample[metric], result.get(metric)):
                raise ValueError(f"historical anchor endpoint mismatch: {record.run_id}: {metric}")
    return {"verified_cells": len(seen), "dataset_digest_exact": True,
            "preprojection_initialization_exact": True, "order_seed_exact": True,
            "contract": "Prior four-way immutable audit certifies optimizer pairing; DA-10 arm independently anchors every new cell."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific-checkout", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    analysis_git = _git_metadata(root)
    _require_reproducible_source("exploratory", analysis_git)
    checkout = args.scientific_checkout
    git = _git_metadata(checkout)
    digest = _source_tree_digest(checkout)["digest"]
    if git.get("revision") != REVISION or git.get("dirty") is not False or digest != DIGEST:
        raise ValueError("publication requires the exact clean no-momentum scientific snapshot")
    config = yaml.safe_load((checkout / "configs/product_unit/manifold_landscape_no_momentum_cpu.yaml").read_text())
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    import wandb
    api = wandb.Api(timeout=120)
    project = credentials["WANDB_ENTITY"] + "/" + credentials["WANDB_PROJECT"]
    cache = root / "wandb/artifacts/punn-manifold-no-momentum-readback"
    checked = check_group(api, project, config, REVISION, DIGEST, cache, full=True, workers=args.workers)
    if checked["missing"] or checked["active"] or len(checked["rows"]) != 480:
        raise ValueError("final publication waits for all 480 outcomes and no active conditions")
    old_artifact = api.artifact(project + "/" + REFERENCE, type="analysis")
    reference = json.loads(download_entry(old_artifact, "comparison_report.json",
                                         root / "wandb/artifacts/punn-manifold-final-comparison-e00u5oe9").read_text())
    old_cache = root / "wandb/artifacts/punn-manifold-final-readback"
    accepted, excluded, _ = collect_completed(api, project, root / ".cache/da10-landscape-cpu-59b53d1", old_cache)
    verify_reference_cache(accepted, reference, project, old_cache, args.workers)
    old_by_cell = {(r.config["task"], r.config["architecture"], r.config["seed"]): r for r in accepted}
    allowed = {"condition_id", "method", "momentum", "device", "landscape_device", "provenance",
               "dataset_artifact", "data_metadata", "wandb_run_id", "wandb_run_url", "run_group", "run_name", "stage"}
    for record in checked["accepted"]:
        old = old_by_cell[(record.config["task"], record.config["architecture"], record.config["seed"])]
        for key, value in record.config.items():
            if key not in allowed and old.config.get(key) != value:
                raise ValueError(f"paired DA-10 recipe mismatch: {record.run_id}: {key}")
    performance_artifact = api.artifact(project + "/" + PERFORMANCE_REFERENCE, type="analysis")
    historical = json.loads(download_entry(performance_artifact, "comparison_report.json",
                                          root / "wandb/artifacts/punn-no-momentum-reference-performance-4761ik87").read_text())
    legacy_config = yaml.safe_load((root / "configs/product_unit/manifold_architecture_stability.yaml").read_text())
    historical_pairing = verify_historical_anchor(historical, records(api, project, legacy_config), checked["rows"])
    from .punn_no_momentum_report import build_report, validate_recorded_pairing, write_report_artifact
    pairing = validate_recorded_pairing(reference["runs"], checked["rows"], old_cache=old_cache,
                                       no_momentum_cache=cache)
    report = build_report(reference, checked, historical_performance=historical)
    report.update(pairing_validation=pairing, historical_pairing=historical_pairing,
                  scientific_revision=REVISION, scientific_source_digest=DIGEST,
                  reference_artifact=old_artifact.qualified_name,
                  historical_performance_artifact=performance_artifact.qualified_name)
    with wandb_analysis_run("da10-no-momentum-final-comparison", "punn-da10-no-momentum-comparison-v1", "analysis",
                            {"scientific_git": git, "scientific_source_digest": digest,
                             "frozen_config": config, "analysis_git": analysis_git,
                             "analysis_source": _source_tree_digest(root), "training_runs_added": 0}) as run:
        artifact = wandb.Artifact("punn-da10-no-momentum-comparison-" + run.id, type="analysis")
        figures = write_report_artifact(report, artifact, old_cache=old_cache, no_momentum_cache=cache)
        run.log_artifact(artifact).wait()
        run.summary.update({"verified_paired_conditions": 480, "analysis_artifact": artifact.qualified_name,
                            "figures": figures, "training_runs_added": 0,
                            "excluded_no_momentum_attempts": len(checked["excluded"]),
                            "excluded_reference_attempts": len(excluded)})
        print(json.dumps({"analysis_run_id": run.id, "url": run.url,
                          "artifact": artifact.qualified_name, "figures": figures}), flush=True)


if __name__ == "__main__":
    main()
