"""
Minimal facade for embedding: :meth:`native` (background) and :meth:`run` (visible terminal).
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional , Any

from src.proj.core import strPath
from .util import process , compose_with_pause , format_python_command , to_shell_string , guess_command_title
from .preference import PAUSE_WHEN_DONE
from .macos import open_in_macos
from .windows import open_for_windows
from .linux import open_in_linux

class Shell:
    """
    Cross-platform terminal launcher.

    - :meth:`run` — run a command in the background without opening a terminal window.
    - :meth:`open` — open the platform default terminal (cmux-first on macOS when enabled) and
      execute a shell line; by default appends a pause before exit.
    - :meth:`run_py` / :meth:`open_py` — convenience wrappers that build the Python command line.
    """
    default_cwd: Optional[strPath] = None

    def __init__(self, *, default_cwd: Optional[strPath] = None) -> None:
        self.set_default_cwd(default_cwd)

    @classmethod
    def set_default_cwd(cls, default_cwd: Optional[strPath]) -> None:
        """Set the class-level default working directory used when ``cwd`` is not explicitly passed."""
        cls.default_cwd = default_cwd

    @staticmethod
    def _resolve_cwd(cwd: Optional[strPath], fallback: Optional[strPath]) -> str:
        """Return an absolute cwd string: prefer ``cwd``, then ``fallback``, then ``Path.cwd()``."""
        base = cwd if cwd is not None else fallback
        if base is not None:
            return str(Path(base).resolve())
        return str(Path.cwd().resolve())

    @classmethod
    def run(
        cls,
        cmd: str | Sequence[str],
        *,
        cwd: Optional[strPath] = None,
        env: Optional[Mapping[str, str]] = None,
    ):
        """
        Start ``cmd`` in the background without attaching a terminal.

        Prefer passing a argv sequence; a string is interpreted by the platform shell
        (``cmd.exe`` on Windows, ``/bin/sh -c`` on Unix).
        """
        workdir = cls._resolve_cwd(cwd, cls.default_cwd)
        return process.spawn_native(cmd, cwd=workdir, env=env)

    @classmethod
    def open(
        cls,
        cmd: str | Sequence[str],
        *,
        pause_when_done: bool = PAUSE_WHEN_DONE ,
        cwd: Optional[strPath] = None ,
        option: Any | None = None ,
        title: str | None = None ,
        new_on: str | None = None ,
        as_workspace: str | None = None ,
        from_workspace: str | None = None ,
    ) -> None:
        """
        Open a terminal and run ``cmd`` (shell line, e.g. output of
        :func:`shell_opener.commands.format_python_command`).

        ``pause_when_done`` defaults to ``True`` so the window stays open for review.
        """
        workdir = cls._resolve_cwd(cwd, cls.default_cwd)
        line = compose_with_pause(to_shell_string(cmd), pause_when_done=pause_when_done)
        if title is None:
            title = guess_command_title(line)
        system = platform.system()
        kwargs = {
            "cwd": workdir,
            "option": option,
            "title": title,
            "new_on": new_on,
            "as_workspace": as_workspace,
            "from_workspace": from_workspace,
        }
        if system == "Darwin":
            open_in_macos(line , **kwargs)
        elif system == "Windows":
            open_for_windows(line, **kwargs)
        elif system == "Linux":
            open_in_linux(line , **kwargs)
        else:
            raise TypeError(f"Unsupported platform {system}")

    @classmethod
    def run_py(
        cls, py_script: strPath, * ,
        py_path: str | None = None, args: Sequence[str] | None = None, kwargs : dict | None = None,
        cwd: Optional[strPath] = None ,
    ) -> None:
        """Run a Python script in the background (no terminal window) via :meth:`run`."""
        cmd = format_python_command(py_script, args=args , kwargs=kwargs , py_path=py_path)
        cls.run(cmd, cwd = cwd)

    @classmethod
    def open_py(
        cls, py_script: strPath, * ,
        py_path: str | None = None, args: Sequence[str] | None = None, kwargs : dict | None = None,
        pause_when_done: bool = PAUSE_WHEN_DONE ,
        cwd: Optional[strPath] = None ,
        option: Any | None = None ,
        title: str | None = None,
        new_on: str | None = None,
        as_workspace: str | None = None ,
        from_workspace: str | None = None ,
    ) -> None:
        """Open a Python script in a visible terminal via :meth:`open`; title defaults to the script filename."""
        if title is None:
            title = Path(py_script).name
        cmd = format_python_command(py_script, args=args , kwargs=kwargs , py_path=py_path)
        cls.open(cmd, pause_when_done = pause_when_done, cwd = cwd, option = option ,
                 title=title, new_on=new_on, as_workspace=as_workspace, from_workspace=from_workspace)

    @classmethod
    def py_cmd(cls, py_script: strPath, * ,
               py_path: str | None = None, args: Sequence[str] | None = None, kwargs : dict | None = None,
               ) -> str:
        """Build and return the shell command line for running a Python script (without launching it)."""
        return format_python_command(py_script, args=args , kwargs=kwargs , py_path=py_path)
