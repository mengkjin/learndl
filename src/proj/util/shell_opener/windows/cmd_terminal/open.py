"""Windows: same pattern as ``example.py`` — ``start cmd /c "…"`` + ``shell=True``."""

from __future__ import annotations

from ...util.process import popen_detached_shell_windows
from .verify import CmdTerminalVerifier

def _cmd_quoted(s: str) -> str:
    """Double-quote for ``cmd.exe`` metasyntax (internal ``"`` → ``""``)."""
    return '"' + s.replace('"', '""') + '"'

class CmdTerminalOpener:
    @classmethod
    def run(cls, cwd: str, command: str) -> None:
        if not CmdTerminalVerifier.available():
            raise RuntimeError("cmd.exe is not available")
        inner = f"cd /d {_cmd_quoted(cwd)} & {command}"
        escaped = inner.replace('"', '""')
        shell_cmd = f'start cmd /c "{escaped}"'
        popen_detached_shell_windows(shell_cmd)