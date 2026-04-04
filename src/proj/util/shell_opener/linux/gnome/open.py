"""Linux: spawn ``gnome-terminal`` in a new window or a new tab."""

from __future__ import annotations

import shlex

from .verify import GnomeTerminalVerifier
from ...preference import LINUX_GNOME_NEW
from ...util import process

class GnomeTerminalOpener:
    @classmethod
    def run(cls, cwd: str, command: str) -> None:
        if not GnomeTerminalVerifier.available():
            raise RuntimeError("gnome-terminal is not available (not on PATH)")
        inner = f"cd {shlex.quote(cwd)} && {command}; exec bash"
        if LINUX_GNOME_NEW == "window":
            process.popen_detached(["gnome-terminal", "--", "bash", "-lc", inner])
        elif LINUX_GNOME_NEW == "tab":
            process.popen_detached(["gnome-terminal", "--tab", "--", "bash", "-lc", inner])
        else:
            raise ValueError(
                f"Invalid LINUX_GNOME_NEW {LINUX_GNOME_NEW!r}; expected 'window' or 'tab'"
            )