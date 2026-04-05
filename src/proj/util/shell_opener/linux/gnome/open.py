"""Linux: spawn ``gnome-terminal`` in a new window or a new tab."""

from __future__ import annotations

import shlex

from .verify import GnomeTerminalVerifier
from ...preference import LINUX_GNOME_NEW
from ...util import process

class GnomeTerminalOpener:
    @classmethod
    def run(cls, command: str, * , cwd: str | None = None, title: str | None = None, new_on: str | None = None) -> None:
        if not GnomeTerminalVerifier.available():
            raise RuntimeError("gnome-terminal is not available (not on PATH)")
        command = f'{command}; exec bash'
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if title is not None:
            command = f'echo -ne "\\033]0;{title}\\a"; {command}'
        if new_on is None:
            new_on = LINUX_GNOME_NEW
        match new_on:
            case "window" | "workspace":
                flag = "--window"
            case "tab":
                flag = "--tab"
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")
        process.popen_detached(["gnome-terminal", flag, "--" , "bash", "-lc", command])