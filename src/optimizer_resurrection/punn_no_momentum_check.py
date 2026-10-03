"""Strict online status and immutable recording checks for the DA-10 ablation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
import yaml

from .punn_architecture_analysis import _terminal_snapshot
from .punn_architecture_stability import MANIFOLD_NO_MOMENTUM_METHOD
from .punn_manifold_landscape_report import (
    audit_runs, download_entry, expected_plan, records, same_snapshot,
)
from .punn_manifold_resume import pending_condition_ids
from .tracking import load_wandb_credentials, require_online_wandb, wandb_analysis_run
from .train import _git_metadata, _source_tree_digest


def check_group(api, project, config, revision, digest, cache, *, full=False,
                engineering=False, workers=4):
    """Count only unique outcomes with checked immutable manifests or raw payloads.

    Active records are validated separately and never accepted as outcomes.
    ``full`` independently re-evaluates retained parameters and landscape grids.
    """
    if config.get("methods") != [MANIFOLD_NO_MOMENTUM_METHOD]:
        raise ValueError("this checker requires the no-momentum DA-10 method")
    if engineering != (config.get("stage") in {"test", "engineering-smoke"}):
        raise ValueError("engineering readback flag does not match the registered stage")
    expected = expected_plan(config)
    observed = records(api, project, config)
    active_records = [r for r in observed if r.state in {"running", "pending", "preempting"}]
    terminal = [r for r in observed if r not in active_records]
    active_ids = set()
    active = []
    for record in active_records:
        condition = record.config.get("condition_id")
        if condition not in expected or condition in active_ids:
            raise ValueError(f"unknown or duplicate active condition: {record.run_id}")
        active_ids.add(condition)
        for key, value in expected[condition].items():
            if record.config.get(key) != value:
                raise ValueError(f"active recipe mismatch: {record.run_id}: {key}")
        if not engineering:
            provenance = record.config.get("provenance", {})
            if (provenance.get("git", {}).get("revision") != revision
                    or provenance.get("git", {}).get("dirty") is not False
                    or provenance.get("source_tree", {}).get("digest") != digest):
                raise ValueError(f"active source mismatch: {record.run_id}")
        active.append({"run_id": record.run_id, "task": record.config["task"],
                       "architecture": record.config["architecture"],
                       "seed": record.config["seed"], "state": record.state,
                       "epoch": record.summary.get("epoch"), "url": record.url})
    if engineering:
        excluded = []
        accepted = []
        seen = set()
        for record in terminal:
            cid = record.config.get("condition_id")
            if cid not in expected:
                raise ValueError(f"unknown engineering condition: {record.run_id}")
            result = _terminal_snapshot(record.summary, record.run_id)
            if (record.state != "finished" or result.get("landscape_recording_complete") is not True
                    or result.get("terminal_outcome") not in {"completed", "numerical_failure"}):
                excluded.append({"run_id": record.run_id, "state": record.state,
                                 "reason": "incomplete_execution_or_recording"})
                continue
            if cid in seen:
                raise ValueError(f"duplicate engineering condition: {cid}")
            for key, value in expected[cid].items():
                if record.config.get(key) != value:
                    raise ValueError(f"engineering recipe mismatch: {record.run_id}: {key}")
            seen.add(cid)
            accepted.append(record)
        missing = set(expected) - seen
    else:
        missing, excluded = pending_condition_ids(expected, terminal, revision, digest)
        accepted = [r for r in terminal if r.state == "finished"
                    and r.summary.get("landscape_recording_complete") is True
                    and _terminal_snapshot(r.summary, r.run_id).get("terminal_outcome")
                    in {"completed", "numerical_failure"}]
    accepted_ids = {r.config["condition_id"] for r in accepted}
    if active_ids & accepted_ids:
        raise ValueError("an active attempt duplicates an accepted scientific outcome")
    outcomes = Counter()
    for record in accepted:
        result = _terminal_snapshot(record.summary, record.run_id)
        if (result.get("method") != MANIFOLD_NO_MOMENTUM_METHOD
                or result.get("momentum") != 0.0
                or result.get("manifold_nesterov") is not False):
            raise ValueError(f"no-momentum terminal recipe mismatch: {record.run_id}")
        provenance = record.config["provenance"]
        if (provenance.get("manifold_nesterov") is not False
                or provenance.get("optimizer_recipe", {}).get("momentum") != 0.0):
            raise ValueError(f"no-momentum optimizer provenance mismatch: {record.run_id}")
        artifact = api.artifact(result["landscape_artifact"], type="loss-landscape")
        entries = artifact.manifest.entries
        if not {"manifest.json", "recorded_landscapes.npz", "provenance.json"} <= set(entries):
            raise ValueError(f"missing retained raw artifact entries: {record.run_id}")
        manifest = json.loads(download_entry(artifact, "manifest.json", cache / "runs" / record.run_id).read_text())
        if (manifest.get("condition_id") != record.config["condition_id"]
                or manifest.get("training_device") != "cpu"
                or manifest.get("landscape_device") != "cpu"):
            raise ValueError(f"immutable manifest identity mismatch: {record.run_id}")
        saved = {k: v for k, v in manifest["terminal_result"].items() if k != "wall_seconds"}
        if not same_snapshot(saved, {k: result.get(k) for k in saved}):
            raise ValueError(f"immutable terminal snapshot mismatch: {record.run_id}")
        if result["terminal_outcome"] == "completed":
            count = config["epochs"] + 2
            schedule = sorted({1, *range(config["slice_every"], config["epochs"] + 1, config["slice_every"])})
            slices = len(schedule) + 2
            if (result.get("recorded_state_count") != count
                    or result.get("slice_count_by_view") != {"ambient": slices, "manifold": slices}):
                raise ValueError(f"incomplete recorded schedule: {record.run_id}")
        outcomes[result["terminal_outcome"]] += 1
    rows = audit_runs(accepted, project, cache, workers) if full else []
    return {"expected": len(expected), "verified_recorded_outcomes": len(accepted),
            "readback": "raw_payload" if full else "immutable_manifest",
            "outcomes": dict(outcomes), "active": active,
            "missing": sorted(missing), "excluded": excluded,
            "accepted": accepted, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific-checkout", type=Path)
    parser.add_argument("--config", default="configs/product_unit/manifold_landscape_no_momentum_cpu.yaml")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    checkout = args.scientific_checkout or root
    config = yaml.safe_load((checkout / args.config).read_text(encoding="utf-8"))
    git = _git_metadata(checkout)
    if not args.engineering and git.get("dirty") is not False:
        raise ValueError("scientific readback requires the clean frozen checkout")
    digest = _source_tree_digest(checkout)["digest"]
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    torch.set_num_threads(1)
    import wandb
    api = wandb.Api(timeout=120)
    project = credentials["WANDB_ENTITY"] + "/" + credentials["WANDB_PROJECT"]
    checked = check_group(api, project, config, git["revision"], digest,
                          root / "wandb/artifacts/punn-manifold-no-momentum-readback",
                          full=args.full, engineering=args.engineering)
    output = {k: v for k, v in checked.items() if k not in {"accepted", "rows"}}
    if args.publish:
        with wandb_analysis_run("da10-no-momentum-recording-readback", "punn-da10-no-momentum-audit-v1",
                                "engineering-smoke" if args.engineering else "analysis",
                                {"scientific_git": git, "source_digest": digest,
                                 "frozen_config": config, "full_readback": args.full}) as run:
            artifact = wandb.Artifact("punn-da10-no-momentum-audit-" + run.id, type="analysis")
            with artifact.new_file("audit_report.json", mode="w", encoding="utf-8") as handle:
                json.dump({**output, "rows": checked["rows"]}, handle, sort_keys=True, allow_nan=False)
            run.log_artifact(artifact).wait()
            run.summary.update(output)
            output.update(audit_run_id=run.id, url=run.url, artifact=artifact.qualified_name)
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
