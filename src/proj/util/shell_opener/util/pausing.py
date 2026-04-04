"""Compose shell lines with optional pause-after snippet."""

from __future__ import annotations

import sys

__all__ = ["compose_with_pause"]

_DEFAULT_PAUSE_UNIX = (
    "echo 'Task complete. Press any key to exit...'; "
    "(read -k 1 2>/dev/null) || read -r -n 1 -s; exit"
)

# cmd.exe: ``timeout /t -1`` waits for a key then returns; with ``cmd /c`` the window closes.
# (``pause`` leaves an interactive shell when combined with ``/k`` / nested ``start``.)
_DEFAULT_PAUSE_WIN = (
    "echo. & echo Task complete. Press any key to exit... & timeout /t -1 >nul"
)

def compose_with_pause(command: str, *, pause_when_done: bool) -> str:
    if not pause_when_done:
        return command
    if sys.platform == "win32":
        return f"{command} & {_DEFAULT_PAUSE_WIN}"
    return f"{command}; {_DEFAULT_PAUSE_UNIX}"
