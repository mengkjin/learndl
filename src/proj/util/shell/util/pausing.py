"""Compose shell lines with optional pause-after snippet."""

from __future__ import annotations

import sys
from typing import Literal

__all__ = ["compose_with_done_action" , "DoneActionType"]

DoneActionType = Literal['pause' , 'close' , 'keep']

_WIN_ACTION_SUFFIX : dict[DoneActionType , str] = {
    'pause': " & echo. & echo Task complete. Press any key to exit... & timeout /t -1 >nul & exit",
    'close': " & exit" ,
    'keep': '',
}
_UNIX_ACTION_SUFFIX : dict[DoneActionType , str] = {
    'pause': " ; echo 'Task complete. Press any key to exit...'; (read -k 1 2>/dev/null) || read -r -n 1 -s; exit" ,
    'close': " ; exit",
    'keep': '',
}

def compose_with_done_action(command: str, *, done_action: DoneActionType = 'pause') -> str:
    """Append a platform-appropriate 'press any key to exit' snippet to ``command`` when requested."""
    if sys.platform == "win32":
        return f"{command}{_WIN_ACTION_SUFFIX[done_action]}"
    else:
        return f"{command}{_UNIX_ACTION_SUFFIX[done_action]}"
