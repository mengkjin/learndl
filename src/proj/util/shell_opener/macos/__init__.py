"""macOS terminal backends: cmux (preferred), Ghostty, or Terminal.app."""

from __future__ import annotations


from typing import Literal

from .cmux import CmuxOpener
from .ghostty import GhosttyOpener
from .terminal_app import TerminalAppOpener
from ..preference import MACOS_OPTIONS

__all__ = ["open_in_macos"]

def get_opener(option: str):
    """Get the opener for the given option"""
    if option not in MACOS_OPTIONS:
        option = MACOS_OPTIONS[0]
    match option:
        case "ghostty":
            return GhosttyOpener()
        case "terminal.app":
            return TerminalAppOpener()
        case 'cmux':
            return CmuxOpener()
        case _:
            raise ValueError(f"Invalid option: {option}")

def open_in_macos(
    command: str,
    * , 
    cwd: str | None = None,
    option : str | None = None,
    title: str | None = None,
    new_on: str | None = None,
    as_workspace: str | None = None,
    from_workspace: str | None = None,
    **kwargs
) -> None:
    """
    Open ``command`` in a visible terminal.

    Backend is ``shell_opener.preference.MACOS_OPTIONS`` by order:
    ``\"cmux\"`` (async IPC + fallbacks), ``\"ghostty\"`` (Ghostty.app), or
    ``None`` / other → Terminal.app.
    """
    options = [option] + MACOS_OPTIONS if option else MACOS_OPTIONS
    for opt in options:
        opener = get_opener(opt)
        if opener:
            break

    opener.run(command, cwd=cwd, title=title, new_on=new_on , as_workspace=as_workspace, from_workspace=from_workspace, **kwargs)