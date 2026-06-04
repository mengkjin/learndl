"""macOS: spawn a shell in WezTerm via ``wezterm cli spawn`` (new tab or new window)."""

from __future__ import annotations

import shlex
import subprocess

from ...preference import MACOS_WEZTERM_NEW
from ...util.basic import BasicOpener
from ...util import process
from .verify import WezTermVerifier


def activate_wezterm() -> bool:
    """
    Bring WezTerm to the foreground (macOS).

    Returns True if ``osascript`` reported success; False if WezTerm is not installed,
    ``osascript`` is missing, or activation failed.
    """
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "WezTerm" to activate'],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


class WezTermOpener(BasicOpener):
    """Open commands in a WezTerm tab or window on macOS via ``wezterm cli spawn``."""

    def available(self) -> bool:
        """Return True if ``wezterm`` is found on ``PATH``."""
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
        """
        Launch ``command`` in WezTerm on macOS.

        ``new_on="tab"`` spawns in the current window (activating WezTerm first);
        ``"window"`` / ``"workspace"`` opens a new WezTerm window.
        The shell line is wrapped as ``bash -lc`` so login-profile variables are available.
        """
        assert self._available, f"{self.__class__.__name__} is not available"
        command = f"{command}; exec bash"
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if title is not None:
            command = f'echo -ne "\\033]0;{title}\\a"; {command}'
        if new_on is None:
            new_on = MACOS_WEZTERM_NEW
        match new_on:
            case "window" | "workspace":
                args: list[str] = ["wezterm", "cli", "spawn", "--new-window"]
            case "tab":
                activate_wezterm()
                args = ["wezterm", "cli", "spawn"]
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")
        if cwd:
            args.extend(["--cwd", cwd])
        args.extend(["--", "bash", "-lc", command])
        process.popen_detached(args)
