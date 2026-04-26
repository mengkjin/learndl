"""
Find running Python processes by script path or task id (optional ``psutil``).

Not part of the minimal :class:`Shell` API; import when you need instance detection.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from src.proj.core import strPath

from ..preference import DISCOVER_WAIT_TIMEOUT , DISCOVER_WAIT_INTERVAL

class ProcessDiscovery:
    """
    Find running Python processes by script path or task id (requires ``psutil``).

    Matched PIDs are returned sorted by process start time (oldest first); use ``[-1]`` for the
    most recently started match.

    When two matches are **parent and child** (both still match the same script/task), the parent is
    dropped so the list favors the process that usually runs user code (e.g. Windows venv launcher
    + real ``python.exe``). If no such pairs exist, ordering is **create_time** only — same idea
    helps on macOS/Linux when a wrapper spawns a child, though double-Python is rarer there than on
    Windows.
    """

    _env_task_key: str = "TASK_ID"

    @classmethod
    def get_psutil(cls):
        if not hasattr(cls, "_psutil"):
            try:
                import psutil as ps
            except ImportError as e:
                raise RuntimeError(
                    "ProcessDiscovery requires psutil; "
                    "pip install 'terminal-opener[discovery]'"
                ) from e
            cls._psutil = ps
        return cls._psutil

    @classmethod
    def find_running_instances(
        cls,
        *,
        script: Optional[strPath] = None,
        task_id: Optional[str] = None,
    ) -> list[int]:
        """
        Return PIDs of Python interpreter processes whose cmdline or environment matches.

        - ``script``: resolved path or its basename appears in argv / cmdline text.
        - ``task_id``: substring in cmdline or process env ``TASK_ID`` equals ``task_id``.

        Results are sorted by ``create_time`` ascending (newest is ``result[-1]``). If another
        matched PID is this PID's parent, the parent is omitted (monitor the child); otherwise
        ordering is unchanged.
        """
        if script is None and task_id is None:
            raise ValueError("Provide script and/or task_id")

        script_resolved: Optional[str] = None
        if script is not None:
            script_resolved = str(Path(script).resolve())

        psutil = cls.get_psutil()
        matches: list[tuple[int, float]] = []

        for proc in psutil.process_iter(
            ["pid", "name", "cmdline", "environ", "create_time"]
        ):
            try:
                info = proc.info
                name = (info.get("name") or "").lower()
                cmdline = info.get("cmdline") or []
                line = " ".join(cmdline)
                env = info.get("environ") or {}

                py_like = "python" in name or any(
                    seg and "python" in seg.lower()
                    for seg in (cmdline[:2] if cmdline else [])
                )
                if not py_like:
                    continue

                if script_resolved is not None:
                    script_ok = (
                        script_resolved in cmdline
                        or script_resolved in line
                        or os.path.basename(script_resolved) in line
                    )
                    if not script_ok:
                        continue

                if task_id is not None:
                    task_ok = task_id in line or env.get(cls._env_task_key) == task_id
                    if not task_ok:
                        continue

                pid = info.get("pid")
                if not isinstance(pid, int):
                    continue

                ct = info.get("create_time")
                if ct is None:
                    try:
                        ct = float(psutil.Process(pid).create_time())
                    except (psutil.Error, OSError, TypeError, ValueError):
                        ct = 0.0
                else:
                    ct = float(ct)

                matches.append((pid, ct))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        matches.sort(key=lambda x: x[1])
        matches = cls._collapse_matched_parents(matches, psutil)
        return [p for p, _ in matches]

    @staticmethod
    def _collapse_matched_parents(
        matches: list[tuple[int, float]],
        ps,
    ) -> list[tuple[int, float]]:
        """If PID and its parent are both matched, drop the parent (prefer child for monitoring)."""
        if len(matches) < 2:
            return matches
        pids = {pid for pid, _ in matches}
        drop_parent: set[int] = set()
        for pid, _ in matches:
            try:
                ppid = int(ps.Process(pid).ppid())
            except (ps.Error, OSError, TypeError, ValueError):
                continue
            if ppid in pids:
                drop_parent.add(ppid)
        if not drop_parent:
            return matches
        kept = [(pid, ct) for pid, ct in matches if pid not in drop_parent]
        kept.sort(key=lambda x: x[1])
        return kept

    @classmethod
    def wait_for_running_instances(
        cls,
        *,
        script: Optional[strPath] = None,
        task_id: Optional[str] = None,
    ) -> list[int]:
        """
        Poll :meth:`find_running_instances` until at least one PID is found or class timeout elapses.

        Use after :meth:`shell_opener.shell.Shell.run`: the interpreter in the new terminal
        usually appears after the call returns.
        """
        deadline = time.monotonic() + DISCOVER_WAIT_TIMEOUT
        time.sleep(DISCOVER_WAIT_INTERVAL)
        while time.monotonic() < deadline:
            hits = cls.find_running_instances(script=script, task_id=task_id)
            if hits:
                return hits
            time.sleep(DISCOVER_WAIT_INTERVAL)
        return []
