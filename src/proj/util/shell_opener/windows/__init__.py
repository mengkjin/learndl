"""Windows terminal backends: WezTerm and cmd.exe, selected by ``WINDOWS_OPTIONS`` preference."""

from ..preference import WINDOWS_OPTIONS
from .cmd_terminal import CmdTerminalOpener
from .wezterm import WezTermOpener

__all__ = ["open_for_windows"]

def get_opener(option: str):
    """Get the opener for the given option"""
    if option not in WINDOWS_OPTIONS:
        option = WINDOWS_OPTIONS[0]
    match option:
        case "cmd":
            return CmdTerminalOpener()
        case "wezterm":
            return WezTermOpener()
        case _:
            raise ValueError(f"Invalid option: {option}")

def open_for_windows(
    command: str,
    *,
    cwd: str | None = None,
    option: str | None = None,
    title: str | None = None,
    new_on: str | None = None,
    **kwargs,
) -> None:
    """
    Open ``command`` in a visible terminal on Windows.

    Backend is ``WINDOWS_OPTIONS`` by order: ``"wezterm"``, ``"cmd"``.
    An explicit ``option`` is tried first, then the ordered list.
    """
    options = [option] + WINDOWS_OPTIONS if option else WINDOWS_OPTIONS
    for opt in options:
        opener = get_opener(opt)
        if opener:
            break
    opener.run(command, cwd=cwd, title=title, new_on=new_on)