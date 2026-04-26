"""Helpers to build shell commands (e.g. run a Python file)."""

from __future__ import annotations

import re
import shlex
import sys
import subprocess
from pathlib import Path
from typing import Sequence, Any

from src.proj.core import strPath

__all__ = ["format_python_command" , "to_shell_string" , "guess_command_title"]

def _win_cmd_quote(s: str) -> str:
    """Quote a token for ``cmd.exe`` (double quotes; internal ``\"\"``)."""
    if not s:
        return '""'
    return '"' + s.replace('"', '""') + '"'


# If these appear in a token, it must be quoted for ``start cmd /c "…"`` / nested cmd lines.
_WIN_CMD_NEED_QUOTE = frozenset(' \t&|^<>()%"')


def _win_cmd_token(s: str) -> str:
    """
    One argument for a ``cmd.exe`` line: omit quotes when safe (typical paths without spaces).

    Extra ``"`` inside ``start cmd /c "…"`` often breaks parsing and makes Python treat
    ``python.exe`` as a ``.py`` file (SyntaxError ``\\x90``).
    """
    if not s:
        return '""'
    if any(ch in s for ch in _WIN_CMD_NEED_QUOTE):
        return _win_cmd_quote(s)
    return s

def to_shell_string(cmd_list : Sequence[Any] | str) -> str:
    """Convert an argv sequence to a properly-quoted shell string, or pass a string through unchanged."""
    if isinstance(cmd_list, str):
        return cmd_list
    if sys.platform == "win32":
        return subprocess.list2cmdline([str(x) for x in cmd_list])
    else:
        return ' '.join(shlex.quote(str(x)) for x in cmd_list)

def format_python_command(
    script: strPath,
    args: Sequence[str] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    py_path: str | None = None,
) -> str:
    """Return a single shell line: ``python script.py arg1 …``.

    On Windows, tokens are only quoted when they contain spaces or cmd metacharacters
    (``&|^<>()`` etc.); bare paths keep ``start cmd /c "…"`` escaping reliable.
    """
    exe = py_path or sys.executable
    script_s = str(Path(script).resolve())
    if sys.platform == "win32":
        if exe == "uv run":
            parts = ["uv", "run", _win_cmd_token(script_s)]
        else:
            parts = [_win_cmd_token(exe), _win_cmd_token(script_s)]
    else:
        if exe == "uv run":
            parts = [exe, shlex.quote(script_s)]
        else:
            parts = [shlex.quote(exe), shlex.quote(script_s)]
    if args:
        if sys.platform == "win32":
            parts.extend(_win_cmd_token(a) for a in args)
        else:
            parts.extend(shlex.quote(a) for a in args)
    if kwargs:
        parts.extend(f"--{k} {str(v).replace(' ', '')}" for k, v in kwargs.items() if str(v).strip())
    return " ".join(parts)

def guess_command_title(command: str) -> str | None:
    """
    extract .py filename from command:
        python3 any/path/name.py
        python.exe any/path/name.py
        uv run any/path/name.py
        python C:\\my folder\\app.py   #support space in path
    return filename (e.g. name.py), return None if not matched
    """
    pattern = re.compile(
        r'(?:python[\d.]*(?:\.exe)?|uv\s+run)\s+(.*?\.py)(?=[\s;]|$)',
        re.IGNORECASE
    )
    match = pattern.search(command)
    if not match:
        return None
    
    full_path = match.group(1)
    normalized = full_path.replace('\\', '/')
    filename = normalized.split('/')[-1]
    return filename