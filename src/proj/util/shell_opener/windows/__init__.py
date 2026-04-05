from typing import Literal
from ..preference import WINDOWS_OPTIONS
from .cmd_terminal import CmdTerminalOpener

__all__ = ["open_for_windows"]

def open_for_windows(
    command: str , * , 
    cwd: str | None = None, option : str | None = None, 
    title: str | None = None, new_on: str | None = None , **kwargs) -> None:
    if option is None:
        option = WINDOWS_OPTIONS[0]
    match option:
        case "cmd":
            Opener = CmdTerminalOpener
        case _:
            Opener = CmdTerminalOpener
    Opener.run(command, cwd=cwd, title=title, new_on=new_on)