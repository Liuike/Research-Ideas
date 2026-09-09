"""Repository-wide pytest setup.

Codex runs under a sandbox Windows account while inheriting Luke's USERNAME and
TEMP variables. Pytest uses USERNAME for its default private directory name,
which makes it collide with Luke's inaccessible ``pytest-of-Luke`` directory.
Use the real process login to keep test artifacts in system temp without sharing
another account's private pytest root.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pytest


def _actual_process_user() -> str:
    try:
        # os.getlogin() reads the Windows process token rather than USERNAME,
        # which is exactly what the Codex sandbox needs. On Unix/Oscar, the UID
        # is stable even when a batch job has no controlling terminal.
        user = os.getlogin() if os.name == "nt" else f"uid-{os.getuid()}"
    except (AttributeError, OSError):
        user = f"pid-owner-{os.getpid()}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", user)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None:
        config.option.basetemp = Path(tempfile.gettempdir()) / (
            f"optimizer-resurrection-pytest-{_actual_process_user()}"
        )
