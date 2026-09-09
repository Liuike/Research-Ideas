from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


WANDB_CREDENTIAL_KEYS = ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    state: str
    config: dict[str, Any]
    summary: dict[str, Any]
    url: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_wandb_credentials(project_root: Path | None = None) -> dict[str, str]:
    """Load only W&B credentials, without overriding the process environment."""
    root = project_root or _project_root()
    secret_path = root / ".secrets" / "env"
    if secret_path.exists():
        for raw_line in secret_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in WANDB_CREDENTIAL_KEYS or os.environ.get(key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                os.environ[key] = value

    missing = [key for key in WANDB_CREDENTIAL_KEYS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            "missing required W&B credentials: " + ", ".join(missing)
        )
    return {key: os.environ[key] for key in WANDB_CREDENTIAL_KEYS}


def require_online_wandb() -> None:
    mode = os.environ.get("WANDB_MODE")
    if mode is not None and mode.lower() != "online":
        raise RuntimeError(
            f"WANDB_MODE={mode!r} is not allowed; experiments require online W&B"
        )


def fetch_wandb_runs(filters: dict[str, Any] | None = None) -> list[RunRecord]:
    """Fetch completed runs from the configured W&B project."""
    credentials = load_wandb_credentials()
    require_online_wandb()
    query = {**(filters or {}), "state": "finished"}
    import wandb

    runs = wandb.Api().runs(
        f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}",
        filters=query,
    )
    return [
        RunRecord(
            run_id=str(run.id),
            state=str(run.state),
            config=dict(run.config),
            summary=dict(run.summary),
            url=str(run.url),
        )
        for run in runs
    ]


@contextmanager
def wandb_analysis_run(
    name: str,
    group: str,
    job_type: str,
    config: dict[str, Any],
) -> Iterator[Any]:
    """Create an online W&B run for an analysis report."""
    credentials = load_wandb_credentials()
    require_online_wandb()
    import wandb

    run = wandb.init(
        project=credentials["WANDB_PROJECT"],
        entity=credentials["WANDB_ENTITY"],
        name=name,
        group=group,
        job_type=job_type,
        config=config,
        mode="online",
        settings=wandb.Settings(mode="online"),
        save_code=False,
    )
    if run is None:
        raise RuntimeError("W&B online initialization returned no analysis run")
    try:
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)
