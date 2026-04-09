"""Linux: spawn ``gnome-terminal`` in a new window or a new tab."""

from __future__ import annotations

import shlex
from .verify import GnomeTerminalVerifier
from ...util.basic import BasicOpener
from ...preference import LINUX_GNOME_NEW
from ...util import process

import subprocess

def raise_recent_terminal() -> bool:
    """
    Finds the most recently active gnome-terminal window, 
    raises it to the top, and returns True. 
    Returns False if no terminal window is found.
    """
    try:
        # -l: list windows, -x: include the WM_CLASS (useful for filtering)
        output = subprocess.check_output(["wmctrl", "-lx"], stderr=subprocess.STDOUT).decode()
        
        # Filter for gnome-terminal windows
        # GNOME Terminal usually has the class 'gnome-terminal-server.Gnome-terminal'
        terminals = [line for line in output.splitlines() if "gnome-terminal" in line.lower()]
        
        if not terminals:
            return False

        # In X11, the LAST window in the list is usually the 
        # highest in the 'stacking order' (most recently used).
        recent_window_id = terminals[-1].split()[0]

        # -i: Interpret window ID as a numeric ID
        # -a: Activate (switch desktop, raise to top, and focus)
        subprocess.run(["wmctrl", "-i", "-a", recent_window_id], check=True)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        # wmctrl not installed or no windows found
        return False


class GnomeTerminalOpener(BasicOpener):
    def available(self) -> bool:
        return GnomeTerminalVerifier.available()

    def run(self, command: str, * , cwd: str | None = None, title: str | None = None, new_on: str | None = None) -> None:
        assert self._available , f"{self.__class__.__name__} is not available"
        command = f'{command}; exec bash'
        
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if title is not None:
            command = f'echo -ne "\\033]0;{title}\\a"; {command}'
        if new_on is None:
            new_on = LINUX_GNOME_NEW
        match new_on:
            case "window" | "workspace":
                flag = "--window"
            case "tab":
                if raise_recent_terminal():
                    flag = "--tab"
                else:
                    flag = "--window"
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")
        process.popen_detached(["gnome-terminal", flag, "--" , "bash", "-lc", command])