"""Strict online readback and publication of the recorded DA-10 landscape study."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numbers
from pathlib import Path
import threading
import time

import numpy as np
import torch
import yaml

from .punn_architecture_analysis import _terminal_snapshot
from .punn_architecture_data import TASK_SPECS
from .punn_manifold_recorded import expand_config, parse_args
from .punn_manifold_resume import pending_continuation_condition_ids
from .tracking import RunRecord, load_wandb_credentials, require_online_wandb, wandb_analysis_run
from .train import _git_metadata, _source_tree_digest

GPU_REV = "c55600e4755adf687753e6dee01b63462bbc6f65"
CPU_REV = "59b53d1ac33b7d8edc836a55f530889296f96746"
GPU_DIGEST = "853afa71ba5f0e4df8f1a78e4967d573c5f476e6ef35522c176e5639768946c6"
CPU_DIGEST = "116c85bc9c09d2cf9e5e02d07b7c657948af206797dafa986fc475cd87a26e3f"
RUNTIME = {"dry_run", "stage", "run_name", "run_group"}


def expected_plan(config):
    return {a.condition_id: {k: v for k, v in vars(a).items() if k not in RUNTIME}
            for a in (parse_args(command[3:]) for command in expand_config(config))}


def records(api, project, config):
    return [RunRecord(r.id, r.state, dict(r.config), dict(r.summary_metrics), r.url)
            for r in api.runs(project, filters={"group": config["run_group"],
                                               "config.protocol": config["protocol"]}, lazy=False)]


def download_entry(artifact, name, directory):
    """Read immutable artifact bytes; reuse only a checksum-verified W&B cache."""
    from wandb.sdk.lib.hashutil import md5_file_b64
    entry = artifact.get_path(name)
    path = directory / name
    if path.exists():
        if md5_file_b64(str(path)) != entry.digest:
            raise ValueError(f"cached artifact checksum mismatch: {artifact.qualified_name}/{name}")
    else:
        import requests
        for attempt in range(3):
            try:
                path = Path(entry.download(root=str(directory)))
                break
            except (requests.RequestException, PermissionError):
                if attempt == 2:
                    raise
                time.sleep(2)
        if md5_file_b64(str(path)) != entry.digest:
            raise ValueError(f"downloaded artifact checksum mismatch: {artifact.qualified_name}/{name}")
    return path


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def metadata(payload):
    return json.loads(str(payload["metadata_json"]))


def same_snapshot(saved, online):
    """Allow JSON numeric rounding; keep structure and nonnumeric values exact."""
    if isinstance(saved, dict):
        return isinstance(online, dict) and set(saved) == set(online) and all(
            same_snapshot(v, online[k]) for k, v in saved.items())
    if isinstance(saved, list):
        return isinstance(online, list) and len(saved) == len(online) and all(
            same_snapshot(a, b) for a, b in zip(saved, online))
    if isinstance(saved, bool) or isinstance(online, bool):
        return type(saved) is type(online) and saved == online
    if isinstance(saved, numbers.Real) and isinstance(online, numbers.Real):
        return bool(np.isclose(saved, online, rtol=1e-10, atol=1e-12))
    return saved == online


def collect_completed(api, project, checkout, cache):
    gpu_config = yaml.safe_load((checkout / "configs/product_unit/manifold_landscape_gpu.yaml").read_text())
    cpu_config = yaml.safe_load((checkout / "configs/product_unit/manifold_landscape_cpu_continuation.yaml").read_text())
    gpu_expected, cpu_expected = expected_plan(gpu_config), expected_plan(cpu_config)
    gpu_records, cpu_records = records(api, project, gpu_config), records(api, project, cpu_config)
    pending, excluded, accepted_gpu = pending_continuation_condition_ids(
        gpu_expected, cpu_expected, gpu_records, cpu_records,
        source_group=gpu_config["run_group"], current_group=cpu_config["run_group"],
        source_revision=GPU_REV, source_tree_digest=GPU_DIGEST,
        current_revision=CPU_REV, current_tree_digest=CPU_DIGEST,
    )
    if pending:
        raise ValueError(f"final analysis requires all 480 recorded outcomes; missing {len(pending)}")
    accepted = [r for r in gpu_records + cpu_records
                if r.state == "finished" and r.summary.get("landscape_recording_complete") is True]
    if len(accepted) != 480 or len(accepted_gpu) != 116:
        raise ValueError("unexpected final hardware split or accepted outcome count")
    transfer = api.artifact(project + "/punn-da10-hardware-transfer-t6eu6glr:v0", type="analysis")
    mapping = json.loads(download_entry(transfer, "transfer.json", cache / "transfer").read_text())
    observed = {r.run_id: _terminal_snapshot(r.summary, r.run_id)["landscape_artifact"]
                for r in gpu_records if r.state == "finished"}
    registered = {r["gpu_run_id"]: r["artifact"] for r in mapping["retained_source_mapping"]}
    if observed != registered:
        raise ValueError("retained GPU outcomes differ from the immutable transfer audit")
    by_cell = {tuple(v[k] for k in ("task", "architecture", "method", "seed", "data_seed")): cid
               for cid, v in cpu_expected.items()}
    for entry in mapping["retained_source_mapping"]:
        if by_cell[tuple(entry["logical_cell"])] != entry["cpu_condition_id"]:
            raise ValueError("transfer audit logical-cell mapping mismatch")
    return accepted, excluded, cpu_config


def audit_runs(accepted, project, cache, workers):
    import wandb
    from .punn_manifold_landscape_analysis import audit_payload
    local = threading.local()

    def audit(record):
        if not hasattr(local, "api"):
            local.api = wandb.Api(timeout=120)
        result = _terminal_snapshot(record.summary, record.run_id)
        artifact = local.api.artifact(result["landscape_artifact"], type="loss-landscape")
        folder = cache / "runs" / record.run_id
        manifest = json.loads(download_entry(artifact, "manifest.json", folder).read_text())
        payload = read_npz(download_entry(artifact, "recorded_landscapes.npz", folder))
        provenance = json.loads(download_entry(artifact, "provenance.json", folder).read_text())
        for saved_result in (manifest["terminal_result"], provenance["terminal_result"]):
            # The summary updates total wall time after uploading recordings;
            # the immutable snapshot retains training/recording time instead.
            saved_scientific = {k: v for k, v in saved_result.items() if k != "wall_seconds"}
            if not same_snapshot(saved_scientific, {k: result.get(k) for k in saved_scientific}):
                raise ValueError(f"artifact terminal snapshot mismatch: {record.run_id}")
        if provenance["config"] != {k: record.config[k] for k in provenance["config"]}:
            raise ValueError(f"artifact resolved config mismatch: {record.run_id}")
        if provenance["provenance"]["git"] != record.config["provenance"]["git"]:
            raise ValueError(f"artifact source revision mismatch: {record.run_id}")
        if provenance["provenance"]["source_tree"] != record.config["provenance"]["source_tree"]:
            raise ValueError(f"artifact source bytes mismatch: {record.run_id}")
        dataset_ref = manifest["dataset_artifact"]
        if not dataset_ref.startswith(project + "/"):
            dataset_ref = project + "/" + dataset_ref
        dataset_artifact = local.api.artifact(dataset_ref, type="dataset")
        dataset = read_npz(download_entry(dataset_artifact, "dataset.npz", folder / "dataset"))
        row = audit_payload(payload, manifest, result, record.config, dataset, check_terminal_corners=True)
        row.update(run_id=record.run_id, url=record.url, artifact=artifact.qualified_name,
                   dataset_artifact=dataset_artifact.qualified_name,
                   source_revision=record.config["provenance"]["git"]["revision"],
                   source_digest=record.config["provenance"]["source_tree"]["digest"],
                   initialization_digest=result.get("initialization_digest"),
                   preprojection_initialization_digest=result.get("preprojection_initialization_digest"),
                   order_seed=result.get("order_seed"))
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(audit, r) for r in accepted]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 24 == 0:
                print(json.dumps({"raw_recorded_payloads_verified": len(rows), "expected": 480}), flush=True)
    return sorted(rows, key=lambda r: (r["condition"]["task"], r["condition"]["architecture"], r["condition"]["seed"]))


def baseline_rows(api, project, root, cache, da10_rows):
    """Pair only the original small-width f1/f4 models and seeds 0--9."""
    from . import punn_recorded
    from .models.product_unit import ProductUnitNetwork
    from torch.nn.utils import vector_to_parameters
    paired = {(r["condition"]["task"], r["condition"]["seed"]): r for r in da10_rows
              if r["condition"]["architecture"] == "small" and r["condition"]["task"] in {"f1", "f4"}}
    rows = []
    for filename, selected_method in [("landscape_recorded_references.yaml", "sgd"),
                                       ("adamw_landscape.yaml", "adamw"),
                                       ("muon_landscape.yaml", "muon_moonlight")]:
        cfg = yaml.safe_load((root / "configs/product_unit" / filename).read_text())
        expected = {a.condition_id: vars(a) for a in
                    (punn_recorded.parse_args(cmd[3:]) for cmd in punn_recorded.expand_config(cfg))}
        selected = [r for r in records(api, project, cfg)
                    if r.config["method"] == selected_method and r.config["seed"] < 10]
        if len(selected) != 20:
            raise ValueError(f"baseline requires all 20 registered attempts: {selected_method}")
        seen = set()
        sources = set()
        for record in selected:
            c = record.config
            result = dict(record.summary)
            accepted_state = record.state == "finished" or (
                record.state == "failed" and result.get("terminal_outcome") == "numerical_failure")
            if (not accepted_state or result.get("landscape_status") != "completed"
                    or c["condition_id"] not in expected or c["condition_id"] in seen):
                raise ValueError(f"incomplete/duplicate baseline: {record.run_id}")
            seen.add(c["condition_id"])
            for key, value in expected[c["condition_id"]].items():
                if key not in RUNTIME and c.get(key) != value:
                    raise ValueError(f"baseline recipe mismatch: {record.run_id}: {key}")
            provenance = c["provenance"]
            if provenance["git"].get("dirty") is not False:
                raise ValueError("baseline scientific source must be clean")
            sources.add((provenance["git"]["revision"], provenance["source_tree"]["digest"]))
            # The original recorded-reference runner stores its terminal
            # result flat; the newer architecture runner uses a nested copy.
            artifact = api.artifact(result["landscape_artifact"], type="loss-landscape")
            folder = cache / "baselines" / record.run_id
            saved_provenance = json.loads(download_entry(artifact, "provenance.json", folder).read_text())
            # W&B config drops empty nested dictionaries (the local Slurm
            # metadata), and appends its own run identity after initialization.
            saved_source = dict(saved_provenance["provenance"])
            online_source = {k: v for k, v in provenance.items() if k != "wandb"}
            saved_source["environment"] = {k: v for k, v in saved_source["environment"].items() if v != {}}
            online_source["environment"] = {k: v for k, v in online_source["environment"].items() if v != {}}
            if saved_source != online_source:
                raise ValueError(f"baseline artifact provenance mismatch: {record.run_id}")
            if saved_provenance["config"] != {k: c[k] for k in saved_provenance["config"]}:
                raise ValueError(f"baseline artifact resolved config mismatch: {record.run_id}")
            saved_result = {k: v for k, v in saved_provenance["result"].items() if k != "wall_seconds"}
            # On old numerical failures W&B retains a preceding logged metric
            # when the terminal summary update supplies None. The immutable
            # artifact result is authoritative for these missing endpoints.
            stale_metrics = [k for k in ("train_mse", "test_mse")
                             if saved_result.get("terminal_outcome") == "numerical_failure"
                             and saved_result.get(k) is None and result.get(k) is not None]
            comparable = {k: v for k, v in saved_result.items() if k not in stale_metrics}
            if not same_snapshot(comparable, {k: result.get(k) for k in comparable}):
                raise ValueError(f"baseline immutable terminal result mismatch: {record.run_id}")
            result.update(saved_result)
            dataset = read_npz(download_entry(artifact, "dataset.npz", folder))
            payload = read_npz(download_entry(artifact, "training_landscapes.npz", folder))
            m = metadata(payload)
            da10 = paired[(c["task"], c["seed"])]
            original = read_npz(cache / "runs" / da10["run_id"] / "recorded_landscapes.npz")
            da10_dataset = read_npz(cache / "runs" / da10["run_id"] / "dataset" / "dataset.npz")
            for key in ("train_x", "train_y", "test_x", "test_y"):
                if not np.array_equal(dataset[key], da10_dataset[key]):
                    raise ValueError(f"baseline/DA10 dataset mismatch: {record.run_id}: {key}")
            if not np.array_equal(payload["directions"], original["directions/ambient"]):
                raise ValueError(f"baseline/DA10 ambient directions mismatch: {record.run_id}")
            if result.get("order_seed") != da10["order_seed"]:
                raise ValueError(f"baseline/DA10 sample order seed mismatch: {record.run_id}")
            if result.get("initialization_digest") != da10["preprojection_initialization_digest"]:
                raise ValueError(f"baseline/DA10 preprojection initialization mismatch: {record.run_id}")
            slices = []
            for index in sorted({key.split("/")[1] for key in payload if key.startswith("slices/")}):
                prefix = "slices/" + index
                grid = payload[prefix + "/sample/cpu_reference_mse"]
                mask = payload[prefix + "/sample/cpu_reference_nonfinite"]
                if grid.shape != (31, 31) or not np.array_equal(mask, ~np.isfinite(grid)):
                    raise ValueError(f"invalid baseline CPU reference grid: {record.run_id}")
                center = float(grid[15, 15])
                finite = grid[np.isfinite(grid)].astype(np.float64)
                epoch = int(m[prefix + "/epoch"])
                kind = m[prefix + "/kind"]
                coords = payload[prefix + "/sample/coordinates_1d"]
                if not np.allclose(coords, np.linspace(-1, 1, 31), rtol=1e-6, atol=1e-7):
                    raise ValueError("baseline slice coordinate/radius mismatch")
                if epoch == 500 and result["terminal_outcome"] == "completed":
                    input_dim, hidden, _, output = TASK_SPECS[c["task"]]
                    model = ProductUnitNetwork(input_dim, hidden, output, input_domain="real_complex", init_bound=1.0)
                    vector_to_parameters(torch.from_numpy(payload[prefix + "/center"].copy()), model.parameters())
                    with torch.no_grad():
                        scalar = float((model(torch.from_numpy(dataset["train_x"])) - torch.from_numpy(dataset["train_y"])).square().mean())
                    if not np.isclose(center, scalar, rtol=1e-4, atol=1e-7) or not np.isclose(center, result["train_mse"], rtol=1e-4, atol=1e-7):
                        raise ValueError(f"baseline terminal center mismatch: {record.run_id}")
                delta = finite - center if np.isfinite(center) else np.array([])
                slices.append({"view": "ambient", "phase": "terminal" if epoch == 500 and kind == "epoch" else ("initialization" if kind == "initial" else "epoch"),
                               "epoch": epoch, "center_mse": center if np.isfinite(center) else None,
                               "absolute_p95_mse_delta": float(np.quantile(np.abs(delta), .95)) if delta.size else None,
                               "signed_p95_mse_delta": float(np.quantile(delta, .95)) if delta.size else None,
                               "actual_displacement_p95": float(np.quantile(np.linalg.norm(np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1), axis=-1), .95)),
                               "mse_finite_count": int(np.isfinite(grid).sum()), "mse_nonfinite_count": int(mask.sum())})
            rows.append({"condition": {"task": c["task"], "architecture": "small", "method": c["method"], "seed": c["seed"], "data_seed": c["data_seed"], "training_device": "cpu"},
                         "run_id": record.run_id, "url": record.url, "artifact": artifact.qualified_name,
                         "terminal_outcome": result["terminal_outcome"],
                         "train_mse": finite_number(result.get("train_mse")), "test_mse": finite_number(result.get("test_mse")),
                         "slices": slices, "pairing": {"dataset_exact": True, "ambient_directions_exact": True, "preprojection_initialization_exact": True, "order_seed_exact": True},
                         "summary_stale_metric_keys": stale_metrics,
                         "source_revision": provenance["git"]["revision"], "source_digest": provenance["source_tree"]["digest"]})
        if len(sources) != 1:
            raise ValueError(f"unregistered mixed baseline sources: {selected_method}")
    return rows


def finite_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("readback workers must be between 1 and 4")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    checkout = root / ".cache/da10-landscape-cpu-59b53d1"
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    import wandb
    project = credentials["WANDB_ENTITY"] + "/" + credentials["WANDB_PROJECT"]
    api = wandb.Api(timeout=120)
    cache = root / "wandb/artifacts/punn-manifold-final-readback"
    accepted, excluded, config = collect_completed(api, project, checkout, cache)
    rows = audit_runs(accepted, project, cache, args.workers)
    baselines = baseline_rows(api, project, root, cache, rows)
    from .punn_manifold_landscape_analysis import aggregate_audits
    from .punn_manifold_landscape_plot import plot_report
    report = {"protocol": "punn-manifold-landscape-comparison-v1", "expected_logical_cells": 480,
              "verified_recorded_outcomes": len(rows), "device_split": dict(Counter(r["condition"]["training_device"] for r in rows)),
              "excluded_attempts": excluded, "runs": rows, "baselines": baselines,
              "aggregate": aggregate_audits(rows),
              "comparison_limits": ["Sampled two-dimensional slices, not full-space flatness.",
                                    "Equal coordinate radius is not equal realized displacement after polar retraction.",
                                    "DA-10 projection changes capacity; f1-small exponent tangent dimension is zero.",
                                    "Training and diagnostics split across 116 CUDA and 364 CPU outcomes.",
                                    "DA-10 CUDA grids and earlier CPU-reference grids can differ by FP32 arithmetic; centers and two terminal corners are independently checked on CPU.",
                                    "Baseline comparison pairs f1/f4 small models and seeds0-9 only; optimizer recipes differ."]}
    with wandb_analysis_run("da10-recorded-landscape-final-comparison", "punn-manifold-landscape-comparison-v1", "analysis",
                            {"frozen_config": config, "training_runs_added": 0, "analysis_git": _git_metadata(root), "analysis_source": _source_tree_digest(root)}) as run:
        artifact = wandb.Artifact("punn-da10-recorded-landscape-comparison-" + run.id, type="analysis")
        with artifact.new_file("comparison_report.json", mode="w", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, allow_nan=False)
        figures = plot_report(report, artifact)
        run.log_artifact(artifact).wait()
        run.summary.update({"verified_recorded_outcomes": 480, "verified_baseline_records": len(baselines),
                            "excluded_attempts": len(excluded), "analysis_artifact": artifact.qualified_name,
                            "device_split": report["device_split"], "figures": figures, "training_runs_added": 0})
        print(json.dumps({"analysis_run_id": run.id, "url": run.url, "artifact": artifact.qualified_name, "figures": figures}), flush=True)


if __name__ == "__main__":
    main()
