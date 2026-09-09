import os
import sys
import types

import pytest

from optimizer_resurrection.tracking import (
    fetch_wandb_runs,
    load_wandb_credentials,
    require_online_wandb,
    wandb_analysis_run,
)


def test_secret_loader_whitelists_wandb_and_preserves_environment(tmp_path, monkeypatch):
    secret_dir = tmp_path / ".secrets"
    secret_dir.mkdir()
    (secret_dir / "env").write_text(
        "WANDB_API_KEY=file-key\n"
        "WANDB_ENTITY=file-entity\n"
        "WANDB_PROJECT=file-project\n"
        "OSCAR_PASSWORD=must-not-load\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WANDB_ENTITY", "environment-entity")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("OSCAR_PASSWORD", raising=False)

    credentials = load_wandb_credentials(tmp_path)

    assert credentials == {
        "WANDB_API_KEY": "file-key",
        "WANDB_ENTITY": "environment-entity",
        "WANDB_PROJECT": "file-project",
    }
    assert "OSCAR_PASSWORD" not in os.environ


def test_fetch_wandb_runs_forces_finished_filter(monkeypatch):
    observed = {}
    remote_run = types.SimpleNamespace(
        id="abc",
        state="finished",
        config={"track": "mlp"},
        summary={"success": True},
        url="https://wandb.invalid/abc",
    )

    class Api:
        def runs(self, path, filters):
            observed.update(path=path, filters=filters)
            return [remote_run]

    fake = types.ModuleType("wandb")
    fake.Api = Api
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_PROJECT", "project")

    records = fetch_wandb_runs({"state": "running", "config.track": "mlp"})

    assert observed == {
        "path": "entity/project",
        "filters": {"state": "finished", "config.track": "mlp"},
    }
    assert records[0].run_id == "abc"
    assert records[0].summary["success"] is True


def test_missing_wandb_credentials_fail_closed(tmp_path, monkeypatch):
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        load_wandb_credentials(tmp_path)


def test_analysis_run_is_online_and_finishes(monkeypatch):
    observed = {}

    class Run:
        def finish(self, exit_code=None):
            observed["exit_code"] = exit_code

    fake = types.ModuleType("wandb")
    fake.Settings = lambda **kwargs: kwargs

    def init(**kwargs):
        observed["init"] = kwargs
        return Run()

    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_PROJECT", "project")

    with wandb_analysis_run("gate-report", "gates", "gate-check", {"gate": "A"}):
        pass

    assert observed["init"]["mode"] == "online"
    assert observed["init"]["settings"]["mode"] == "online"
    assert observed["init"]["job_type"] == "gate-check"
    assert "dir" not in observed["init"]
    assert observed["exit_code"] == 0


def test_offline_wandb_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(RuntimeError, match="require online W&B"):
        require_online_wandb()
