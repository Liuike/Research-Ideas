"""Online W&B controller for the durable local DA-10 landscape plan.

The experiment planner remains the only training dispatcher. Its stdout is
streamed through this controller so W&B owns the launcher logs as well as each
scientific child's metrics and landscape artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
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
        raise RuntimeError("this local controller requires the Windows named-mutex launcher")
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


@contextmanager
def _exclusive_registered_sweep(credentials, config):
    """Keep the source-group mutex during a hardware continuation as well."""
    prefix = f"{credentials['WANDB_ENTITY']}/{credentials['WANDB_PROJECT']}/"
    groups = [config["run_group"]]
    if "continuation" in config:
        groups.insert(0, config["continuation"]["source_group"])
    with ExitStack() as stack:
        for group in groups:
            stack.enter_context(_exclusive_local_sweep(prefix + group))
        yield


def _normalized_windows_path(value: str | Path | None) -> str:
    return ntpath.normcase(ntpath.normpath(str(value or ""))).rstrip("\\")


def _is_same_invocation_wrapper(
    process: dict[str, object], *, parent_pid: int, wrapper_path: Path,
    expected_arguments: str,
) -> bool:
    """Match only this process's immediate UV venv-Python redirector."""
    try:
        if int(process.get("ProcessId", -1)) != parent_pid:
            return False
    except (TypeError, ValueError):
        return False
    if _normalized_windows_path(process.get("ExecutablePath")) != _normalized_windows_path(wrapper_path):
        return False
    command_line = process.get("CommandLine")
    if not isinstance(command_line, str):
        return False
    match = re.fullmatch(r'\s*(?:"([^"]+)"|(\S+))(?:\s+(.*))?\s*', command_line)
    if match is None:
        return False
    return (match.group(3) or "") == expected_arguments


def _assert_no_existing_workers(config_path: Path, group: str, uv_path: Path):
    """Fail closed if an orphan planner/worker remains after a controller exit."""
    original_argv = getattr(sys, "orig_argv", None)
    expected_arguments = (
        subprocess.list2cmdline(original_argv[1:])
        if isinstance(original_argv, list) and len(original_argv) > 1
        else ""
    )
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
} | Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine)
ConvertTo-Json -InputObject $taskMatches -Compress
"""
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            env=env, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("cannot verify local DA-10 process state; inspect it before dispatch")
    matches = json.loads(result.stdout.strip())
    if isinstance(matches, dict):
        matches = [matches]
    # UV's Windows Python redirector can remain as the immediate parent of
    # this controller. Exempt it only when both its executable and complete
    # argument tail prove that it launched this exact invocation.
    parent_pid = os.getppid()
    wrapper_path = Path(uv_path).with_name("python.exe")
    matches = [
        process for process in matches
        if int(process.get("ProcessId", -1)) != os.getpid()
        and not _is_same_invocation_wrapper(
            process, parent_pid=parent_pid, wrapper_path=wrapper_path,
            expected_arguments=expected_arguments,
        )
    ]
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
    if config.get("device") not in {"cpu", "cuda"} or config.get("landscape_device") != config["device"]:
        raise ValueError("the scientific controller requires matching CPU or CUDA devices")
    git = _git_metadata(root)
    _require_reproducible_source(config["stage"], git)
    credentials = load_wandb_credentials(root)
    require_online_wandb()
    from .experiment import expand_config

    expected = len(expand_config(config))
    command = [str(args.uv.resolve()), "run", "--frozen", "--no-sync", "python",
               "-m", "optimizer_resurrection.experiment", "plan", str(args.config),
               "--run", "--workers", str(args.workers)]
    if args.resume or "continuation" in config:
        command.append("--resume")
    import wandb
    with _exclusive_registered_sweep(credentials, config):
        _assert_no_existing_workers(args.config, config["run_group"], args.uv)
        if "continuation" in config:
            source = config["continuation"]
            _assert_no_existing_workers(root / source["source_config"], source["source_group"], args.uv)
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
        name=f"da10-{config['device']}-landscape-controller", group=config["run_group"] + "-controller",
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
