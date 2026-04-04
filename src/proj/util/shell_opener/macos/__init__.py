"""macOS terminal backends: cmux (preferred), Ghostty, or Terminal.app."""

from __future__ import annotations


from typing import Literal

from .cmux import CmuxOpener
from .ghostty import GhosttyOpener
from .terminal_app import TerminalAppOpener
from ..preference import MACOS_OPTIONS

__all__ = ["open_in_macos"]

def open_in_macos(
    cwd: str,
    command: str,
    * , 
    option : Literal["cmux", "ghostty", "terminal.app"] | None = None,
) -> None:
    """
    Open ``command`` in a visible terminal.

    Backend is ``shell_opener.preference.MACOS_OPTIONS`` by order:
    ``\"cmux\"`` (async IPC + fallbacks), ``\"ghostty\"`` (Ghostty.app), or
    ``None`` / other → Terminal.app.
    """
    if option is None:
        option = MACOS_OPTIONS[0] # type: ignore
    match option:
        case "cmux":
            CmuxOpener.run(cwd, command)
        case "ghostty":
            GhosttyOpener.run(cwd, command)
        case "terminal.app":
            TerminalAppOpener.run(cwd, command)
        case _:
            raise ValueError(f"Unsupported macOS opener: {option}")

