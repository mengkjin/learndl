"""Linux terminal backends: WezTerm and GNOME Terminal, selected by ``LINUX_OPTIONS`` preference."""

from .gnome import GnomeTerminalOpener
from .wezterm import WezTermOpener
from ..preference import LINUX_OPTIONS

__all__ = ["open_in_linux"]

def get_opener(option: str):
    """Get the opener for the given option"""
    if option not in LINUX_OPTIONS:
        option = LINUX_OPTIONS[0]
    match option:
        case "wezterm":
            return WezTermOpener()
        case "gnome":
            return GnomeTerminalOpener()
        case _:
            raise ValueError(f"Invalid option: {option}")

def open_in_linux(
    command: str, * ,
    cwd: str | None = None,  option: str | None = None,
    title: str | None = None, new_on: str | None = None , **kwargs) -> None:
    """
    Open ``command`` in a visible terminal on Linux.

    Backend is ``LINUX_OPTIONS`` by order: ``"wezterm"``, ``"gnome"``.
    An explicit ``option`` is tried first, then the ordered list.
    """
    options = [option] + LINUX_OPTIONS if option else LINUX_OPTIONS
    for opt in options:
        opener = get_opener(opt)
        if opener:
            break
    opener.run(command, cwd=cwd, title=title, new_on=new_on, **kwargs)