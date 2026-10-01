"""Online W&B controller for the durable local DA-10 GPU landscape plan.

The experiment planner remains the only training dispatcher. Its stdout is
streamed through this controller so W&B owns the launcher logs as well as each
scientific child's metrics and landscape artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from .tracking import load_wandb_credentials, require_online_wandb
from .train import _git_metadata, _require_reproducible_source, _source_tree_digest


@contextmanager
def _exclusive_local_sweep(identity: str):
    """Hold an OS mutex across dispatch, closing the two-launch startup race."""
    if os.name != "nt":
        raise RuntimeError("this local GPU controller requires the Windows named-mutex launcher")
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    name = "Local\\OptimizerResurrectionDA10-" + hashlib.sha256(identity.encode()).hexdigest()
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        if status not in {0, 0x80}:
            raise RuntimeError("a local controller already owns this DA-10 sweep")
        acquired = True
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def _assert_no_existing_workers(config_path: Path, group: str):
    """Fail closed if an orphan planner/worker remains after a controller exit."""
    env = dict(os.environ)
    env.update(DA10_CHECK_PID=str(os.getpid()), DA10_CHECK_GROUP=group,
               DA10_CHECK_CONFIG=config_path.name)
    script = r"""
$ErrorActionPreference = 'Stop'
$taskMatches = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.ProcessId -ne [int]$env:DA10_CHECK_PID -and $_.CommandLine -and (
        ($_.CommandLine.Contains('optimizer_resurrection.punn_manifold_recorded') -and
         $_.CommandLine.Contains($env:DA10_CHECK_GROUP)) -or
        ($_.CommandLine.Contains('optimizer_resurrection.experiment') -and
         $_.CommandLine.Contains($env:DA10_CHECK_CONFIG)) -or
        ($_.CommandLine.Contains('optimizer_resurrection.punn_manifold_sweep') -and
         $_.CommandLine.Contains($env:DA10_CHECK_CONFIG))
    )
} | Select-Object ProcessId, ParentProcessId)
ConvertTo-Json -InputObject $taskMatches -Compress
"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            env=env, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("cannot verify local DA-10 process state; inspect it before dispatch")
    matches = json.loads(result.stdout.strip())
    if matches:
        raise RuntimeError(f"existing DA-10 planner/worker processes prevent dispatch: {matches}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("protocol") != "punn-manifold-recorded-v1":
        raise ValueError("controller only supports the registered DA-10 landscape protocol")
    if config.get("device") != "cuda" or config.get("landscape_device") != "cuda":
        raise ValueError("the scientific controller requires CUDA training and landscapes")
    git = _git_metadata(root)
    _require_reproducible_source(config["stage"], git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    from .experiment import expand_config

    expected = len(expand_config(config))
    command = [str(args.uv.resolve()), "run", "--frozen", "--no-sync", "python",
               "-m", "optimizer_resurrection.experiment", "plan", str(args.config),
               "--run", "--workers", str(args.workers)]
    if args.resume:
        command.append("--resume")
    import wandb
    identity = f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}/{config['run_group']}"
    with _exclusive_local_sweep(identity):
        _assert_no_existing_workers(args.config, config["run_group"])
        if not args.resume:
            existing = list(wandb.Api(timeout=90).runs(
                f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}",
                filters={"group": config["run_group"], "config.protocol": config["protocol"]}))
            if existing:
                raise RuntimeError("this group already has attempts; inspect W&B and use explicit --resume")
        return _dispatch(args, root, config, git, credentials, expected, command, wandb)


def _dispatch(args, root, config, git, credentials, expected, command, wandb):
    run = wandb.init(
        project=credentials["WANDB_PROJECT"], entity=credentials["WANDB_ENTITY"],
        name="da10-gpu-landscape-controller", group=config["run_group"] + "-controller",
        job_type="sweep-controller", mode="online", save_code=False,
        config={"protocol": "punn-manifold-landscape-controller-v1", "frozen_config": config,
                "scientific_group": config["run_group"], "expected_runs": expected,
                "workers": args.workers, "command": command, "pid": os.getpid(),
                "source_checkout": str(root), "git": git,
                "source_tree": _source_tree_digest(root)},
    )
    if run is None:
        raise RuntimeError("W&B did not create an online controller")
    started = time.monotonic()
    child_env = dict(os.environ)
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_NAME", "WANDB_RUN_GROUP"):
        child_env.pop(key, None)
    try:
        print(f"CONTROLLER {run.url} | {expected} registered runs | {args.workers} workers", flush=True)
        process = subprocess.Popen(command, cwd=root, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        run.summary.update({"planner_pid": process.pid, "controller_outcome": "running"})
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        exit_code = process.wait()
        run.summary.update({"planner_exit_code": exit_code,
                            "controller_outcome": "completed" if exit_code == 0 else "execution_failure",
                            "wall_seconds": time.monotonic() - started})
        run.finish(exit_code=exit_code)
        return exit_code
    except BaseException:
        if "process" in locals() and process.poll() is None:
            # Terminate only the planner process tree owned by this controller.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        run.summary.update({"controller_outcome": "controller_failure",
                            "wall_seconds": time.monotonic() - started})
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    sys.exit(main())
