from typing import Literal
from terminal_opener.preference import WINDOWS_OPTIONS

from .cmd_terminal import CmdTerminalOpener

__all__ = ["open_for_windows"]

def open_for_windows(cwd: str, command: str , * , option : Literal["cmd"] | None = None) -> None:
    if option is None:
        option = WINDOWS_OPTIONS[0]
    match option:
        case "cmd":
            CmdTerminalOpener.run(cwd, command)
        case _:
            raise ValueError(f"Invalid Windows opener: {option}")