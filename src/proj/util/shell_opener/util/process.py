"""Detached subprocess helpers (no terminal window)."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

Cmd = Union[str, Sequence[str]]

def popen_detached(
    args: list[str],
    *,
    env: Optional[dict[str, str]] = None,
) -> subprocess.Popen:
    kwargs: dict = {
        "args": args,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0x00000008
        )
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True
    if env is not None:
        kwargs["env"] = env
    return subprocess.Popen(**kwargs)


def popen_detached_shell_windows(cmd_line: str, *, env: Optional[dict[str, str]] = None) -> subprocess.Popen:
    """
    Windows: ``Popen(cmd_line, shell=True, …)`` like example.py — used for
    ``start cmd /c "…"`` so ``start`` opens a real console (not a bare ``CREATE_NEW_CONSOLE`` child).
    """
    kwargs: dict = {
        "args": cmd_line,
        "shell": True,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0x00000008),
        "close_fds": True,
    }
    if env is not None:
        kwargs["env"] = env
    return subprocess.Popen(**kwargs)


def spawn_native(
    cmd: Cmd,
    *,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.Popen:
    """
    Run ``cmd`` in the background without opening a terminal window.
    Pass a argv sequence for predictable parsing; a string uses ``cmd.exe /c`` on Windows
    and ``/bin/sh -c`` elsewhere.
    """
    use_env = dict(os.environ)
    if env:
        use_env.update(env)
    cwd_s = str(Path(cwd).resolve()) if cwd is not None else None
    kwargs: dict = {
        "cwd": cwd_s,
        "env": use_env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    else:
        kwargs["start_new_session"] = True

    if isinstance(cmd, str):
        if sys.platform == "win32":
            return subprocess.Popen(cmd, shell=True , **kwargs)
        else:
            return subprocess.Popen(["/bin/sh", "-c", cmd], **kwargs)
    else:
        kwargs['args'] = [str(x) for x in cmd]
        if sys.platform == "win32":
            kwargs["close_fds"] = True
        return subprocess.Popen(**kwargs)
