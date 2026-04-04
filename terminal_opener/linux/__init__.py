from typing import Literal
from terminal_opener.preference import LINUX_OPTIONS
from .gnome import GnomeTerminalOpener

__all__ = ["open_in_linux"]

def open_in_linux(cwd: str, command: str, * , option: Literal["gnome"] | None = None) -> None:
    if option is None:
        option = LINUX_OPTIONS[0]
    match option:
        case "gnome":
            GnomeTerminalOpener.run(cwd, command)
        case _:
            raise ValueError(f"Invalid Linux opener: {option}")