from typing import Literal

from .gnome import GnomeTerminalOpener
from ..preference import LINUX_OPTIONS

__all__ = ["open_in_linux"]

def open_in_linux(
    command: str, * , 
    cwd: str | None = None,  option: str | None = None, 
    title: str | None = None, new_on: str | None = None , **kwargs) -> None:
    if option is None:
        option = LINUX_OPTIONS[0]
    match option:
        case "gnome":
            Opener = GnomeTerminalOpener
        case _:
            Opener = GnomeTerminalOpener

    Opener.run(command, cwd=cwd, title=title, new_on=new_on)