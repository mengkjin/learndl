"""Session control signals, git watching, and DirectCall restart helpers."""
from __future__ import annotations

import os
import subprocess
from typing import Any

__all__ = [
    'ProcessMagicInput',
    'ProcessReload',
    'ProcessSpawn',
    'ProcessSpawnDown',
    'ProcessQuit',
    'git_head',
    'GitHeadWatcher',
    'can_exec_restart',
    'build_direct_call_script',
    'build_exec_argv',
]


class ProcessMagicInput(Exception):
    """Base signal for magic commands that alter the running DirectCall session."""

    def __init__(self, reason: str = '') -> None:
        self.reason = reason
        super().__init__(reason)


class ProcessReload(ProcessMagicInput):
    """Signal that the current process should exec-restart to pick up new code."""

class ProcessSpawn(ProcessMagicInput):
    """Signal that the same DirectCall should open in a new terminal pane."""

class ProcessSpawnDown(ProcessMagicInput):
    """Signal that the same DirectCall should open in a new terminal pane below."""

class ProcessQuit(ProcessMagicInput):
    """Signal that the current process should exit cleanly."""


def git_head() -> str | None:
    """Return current git HEAD hash, or None if unavailable."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


class GitHeadWatcher:
    """Track git HEAD; report when it changes from the baseline captured at init."""

    def __init__(self) -> None:
        self._baseline = git_head()

    @property
    def active(self) -> bool:
        return self._baseline is not None

    def changed(self) -> bool:
        if self._baseline is None:
            return False
        current = git_head()
        return current is not None and current != self._baseline


def can_exec_restart() -> bool:
    """Return False when exec/spawn would kill a Streamlit server process."""
    return not os.environ.get('STREAMLIT_SERVER_PORT')


def build_direct_call_script(cls: type[Any], kwargs: dict[str, Any]) -> str:
    """Build the ``python -c`` script body for a DirectCall invocation."""
    kw_repr = ', '.join(f'{key}={value!r}' for key, value in kwargs.items())
    return f'from {cls.__module__} import {cls.__name__}; {cls.__name__}.go({kw_repr})'


def build_exec_argv(cls: type[Any], kwargs: dict[str, Any]) -> list[str]:
    """Build argv for ``os.execvp`` to restart a DirectCall with the same kwargs."""
    return ['uv', 'run', '--frozen', 'python', '-c', build_direct_call_script(cls, kwargs)]
