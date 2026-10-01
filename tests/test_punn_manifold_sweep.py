from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid

import pytest

from optimizer_resurrection import punn_manifold_sweep as sweep
from optimizer_resurrection.punn_manifold_sweep import _exclusive_local_sweep


@pytest.mark.skipif(os.name != "nt", reason="local Windows controller")
def test_controller_mutex_rejects_concurrent_owner_and_releases_afterward():
    identity = "test-" + uuid.uuid4().hex

    def contend():
        with pytest.raises(RuntimeError, match="already owns"):
            with _exclusive_local_sweep(identity):
                pass

    with _exclusive_local_sweep(identity):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(contend).result(timeout=10)
    with _exclusive_local_sweep(identity):
        pass


_CONTROLLER_ARGS = [
    "-m", "optimizer_resurrection.punn_manifold_sweep",
    "--config", r"C:\work\configs\product_unit\manifold_landscape_gpu.yaml",
    "--workers", "4",
    "--uv", r"C:\work\.venv\Scripts\uv.exe",
]
_CONTROLLER_PID = 50820
_REDIRECTOR_PID = 6392
_UV_PID = 7000
_CONFIG = Path(_CONTROLLER_ARGS[3])
_UV = Path(_CONTROLLER_ARGS[-1])
_GROUP = "punn-manifold-landscape-gpu-v1"


def _process(pid, parent_pid, executable, args):
    return {
        "ProcessId": pid,
        "ParentProcessId": parent_pid,
        "ExecutablePath": executable,
        "CommandLine": f'"{executable}" {subprocess.list2cmdline(args)}',
    }


def _mock_process_snapshot(monkeypatch, processes):
    monkeypatch.setattr(sweep.os, "getpid", lambda: _CONTROLLER_PID)
    monkeypatch.setattr(sweep.os, "getppid", lambda: _REDIRECTOR_PID)
    monkeypatch.setattr(
        sweep.sys,
        "orig_argv",
        [r"C:\Python\python.exe", *_CONTROLLER_ARGS],
        raising=False,
    )

    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(processes))

    monkeypatch.setattr(sweep.subprocess, "run", run)


def _own_process_chain(*, wrapper_executable=None, wrapper_args=None):
    wrapper_executable = wrapper_executable or r"C:\work\.venv\Scripts\python.exe"
    wrapper_args = _CONTROLLER_ARGS if wrapper_args is None else wrapper_args
    return [
        _process(_CONTROLLER_PID, _REDIRECTOR_PID, r"C:\Python\python.exe", _CONTROLLER_ARGS),
        _process(_REDIRECTOR_PID, _UV_PID, wrapper_executable, wrapper_args),
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows process guard")
def test_process_guard_ignores_only_matching_immediate_uv_python_wrapper(monkeypatch):
    _mock_process_snapshot(monkeypatch, _own_process_chain())

    sweep._assert_no_existing_workers(_CONFIG, _GROUP, _UV)


@pytest.mark.parametrize(
    "kind",
    ["second_controller", "planner", "worker", "wrong_wrapper_executable", "wrong_wrapper_args"],
)
@pytest.mark.skipif(os.name != "nt", reason="Windows process guard")
def test_process_guard_still_rejects_unrelated_or_mismatched_processes(monkeypatch, kind):
    processes = _own_process_chain()
    if kind == "second_controller":
        other = _process(8000, 2000, r"C:\Python\python.exe", _CONTROLLER_ARGS)
    elif kind == "planner":
        other = _process(
            8000, 2000, r"C:\Python\python.exe",
            ["-m", "optimizer_resurrection.experiment", "plan", str(_CONFIG)],
        )
    elif kind == "worker":
        other = _process(
            8000, 2000, r"C:\Python\python.exe",
            ["-m", "optimizer_resurrection.punn_manifold_recorded",
             "--run-group", _GROUP],
        )
    elif kind == "wrong_wrapper_executable":
        processes[1] = _process(
            _REDIRECTOR_PID, _UV_PID, r"C:\another-venv\Scripts\python.exe", _CONTROLLER_ARGS
        )
    else:
        wrong_args = [*_CONTROLLER_ARGS[:-3], "3", *_CONTROLLER_ARGS[-2:]]
        processes[1] = _process(
            _REDIRECTOR_PID, _UV_PID, r"C:\work\.venv\Scripts\python.exe", wrong_args
        )
    if kind in {"second_controller", "planner", "worker"}:
        processes.append(other)
    _mock_process_snapshot(monkeypatch, processes)

    with pytest.raises(RuntimeError, match="existing DA-10 planner/worker processes"):
        sweep._assert_no_existing_workers(_CONFIG, _GROUP, _UV)
