"""Reconcile online DA-10 landscape attempts before a recorded-plan resume."""

from __future__ import annotations

from typing import Any

from .punn_architecture_analysis import _terminal_snapshot


def pending_condition_ids(expected: dict[str, dict[str, Any]], records: list[Any],
                          source_revision: str, source_tree_digest: str | None = None,
                          ) -> tuple[set[str], list[dict[str, Any]]]:
    """Accept unique uploaded scientific outcomes; reject overlapping active jobs."""
    accepted: set[str] = set()
    excluded: list[dict[str, Any]] = []
    for record in records:
        condition = record.config.get("condition_id")
        if condition not in expected:
            raise ValueError(f"unregistered landscape condition: {record.run_id}")
        if record.state in {"running", "pending", "preempting"}:
            raise ValueError(f"active landscape run prevents a duplicate resume: {record.run_id}")
        try:
            result = _terminal_snapshot(record.summary, record.run_id)
        except ValueError:
            result = {}
        if (record.state != "finished"
                or result.get("terminal_outcome") not in {"completed", "numerical_failure"}
                or result.get("landscape_recording_complete") is not True):
            excluded.append({"run_id": record.run_id, "condition_id": condition,
                             "state": record.state, "reason": "incomplete_execution_or_recording"})
            continue
        for key, value in expected[condition].items():
            if record.config.get(key) != value:
                raise ValueError(f"landscape recipe mismatch: {record.run_id}: {key}")
        git = record.config.get("provenance", {}).get("git", {})
        if git.get("revision") != source_revision or git.get("dirty") is not False:
            raise ValueError(f"landscape resume requires the original clean source: {record.run_id}")
        if source_tree_digest is not None:
            observed_digest = record.config.get("provenance", {}).get("source_tree", {}).get("digest")
            if observed_digest != source_tree_digest:
                raise ValueError(f"landscape resume requires the original source bytes: {record.run_id}")
        if result.get("terminal_outcome") == "completed":
            if result.get("epochs_completed") != expected[condition]["epochs"]:
                raise ValueError(f"incomplete final epoch: {record.run_id}")
        if not isinstance(result.get("landscape_artifact"), str) or ":v" not in result["landscape_artifact"]:
            raise ValueError(f"missing immutable landscape artifact: {record.run_id}")
        if condition in accepted:
            raise ValueError(f"duplicate accepted landscape condition: {condition}")
        accepted.add(condition)
    return set(expected) - accepted, excluded


def select_pending_commands(commands: list[list[str]], config: dict[str, Any]) -> list[list[str]]:
    import wandb
    from pathlib import Path
    from .punn_manifold_recorded import parse_args
    from .tracking import RunRecord, load_wandb_credentials, require_online_wandb
    from .train import _git_metadata, _require_reproducible_source
    from .train import _source_tree_digest

    root = Path(__file__).resolve().parents[2]
    git = _git_metadata(root)
    _require_reproducible_source(config["stage"], git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    query = {"group": config["run_group"], "config.protocol": config["protocol"]}
    records = [RunRecord(str(r.id), str(r.state), dict(r.config), dict(r.summary), str(r.url))
               for r in wandb.Api(timeout=90).runs(
                   f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}", filters=query)]
    parsed = [(command, vars(parse_args(command[3:]))) for command in commands]
    runtime = {"dry_run", "stage", "run_name", "run_group"}
    expected = {values["condition_id"]: {k: v for k, v in values.items() if k not in runtime}
                for _command, values in parsed}
    pending, excluded = pending_condition_ids(
        expected, records, git["revision"], _source_tree_digest(root)["digest"])
    # Disclose incomplete infrastructure attempts rather than overwriting them.
    import json
    print(json.dumps({"resume_pending": len(pending), "resume_accepted": len(expected)-len(pending),
                      "excluded_incomplete_attempts": excluded}), flush=True)
    return [command for command, values in parsed if values["condition_id"] in pending]
