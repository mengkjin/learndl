"""Linux: spawn a shell in WezTerm via ``wezterm cli spawn`` (new tab or new window)."""

from __future__ import annotations

import shlex
import subprocess

from ...preference import LINUX_WEZTERM_NEW
from ...util import process
from ...util.basic import BasicOpener
from .verify import WezTermVerifier


def activate_wezterm() -> bool:
    """
    Bring WezTerm to the foreground (Linux, X11 via ``wmctrl``).

    Returns True if a WezTerm window was found and activated; False if ``wmctrl`` is
    missing, no matching window exists, or activation failed.
    """
    try:
        output = subprocess.check_output(["wmctrl", "-lx"], stderr=subprocess.STDOUT).decode()
        lines = [
            line
            for line in output.splitlines()
            if "wezterm" in line.lower()
        ]
        if not lines:
            return False
        window_id = lines[-1].split()[0]
        subprocess.run(
            ["wmctrl", "-i", "-a", window_id],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return False


class WezTermOpener(BasicOpener):
    def available(self) -> bool:
        return WezTermVerifier.available()

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        title: str | None = None,
        new_on: str | None = None,
        **kwargs,
    ) -> None:
        assert self._available, f"{self.__class__.__name__} is not available"
        command = f"{command}; exec bash"
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if title is not None:
            command = f'echo -ne "\\033]0;{title}\\a"; {command}'
        if new_on is None:
            new_on = LINUX_WEZTERM_NEW
        match new_on:
            case "window" | "workspace":
                args: list[str] = ["wezterm", "start", "--new-window"]
            case "tab":
                if activate_wezterm():
                    args = ["wezterm", "cli", "spawn"]
                else:
                    args = ["wezterm", "start", "--new-window"]
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")
        if cwd:
            args.extend(["--cwd", cwd])
        args.extend(["--", "bash", "-lc", command])
        process.popen_detached(args)
