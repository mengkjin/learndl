"""Windows: same pattern as ``example.py`` — ``start cmd /c "…"`` + ``shell=True``."""

from __future__ import annotations

from ...util.process import popen_detached_shell_windows
from ...util.basic import BasicOpener
from .verify import CmdTerminalVerifier


def _cmd_quoted(s: str) -> str:
    """Double-quote for ``cmd.exe`` metasyntax (internal ``"`` → ``""``)."""
    return '"' + s.replace('"', '""') + '"'


class CmdTerminalOpener(BasicOpener):
    """Open commands in a new cmd.exe console window on Windows via ``start cmd /c "…"``."""

    def available(self) -> bool:
        """Return True if ``cmd.exe`` is reachable on this Windows system."""
        return CmdTerminalVerifier.available()

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        title: str | None = None,
        new_on: str | None = None,
    ) -> None:
        """
        Launch ``command`` in a new cmd.exe window.

        If ``cwd`` is provided, prepends ``cd /d <cwd> &`` before the command.
        Uses ``shell=True`` + ``DETACHED_PROCESS`` so the window is independent of the parent.
        """
        assert self._available, f"{self.__class__.__name__} is not available"
        if cwd:
            command = f"cd /d {_cmd_quoted(cwd)} & {command}"
        escaped = command.replace('"', '""')
        shell_cmd = f'start cmd /c "{escaped}"'
        popen_detached_shell_windows(shell_cmd)
