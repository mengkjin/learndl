"""cmux CLI presence and socket reachability."""

from __future__ import annotations

import subprocess
import json

from typing import Any

# from src.proj.util.shell_opener.util.commands import to_shell_string

__all__ = ["CmuxCli"]

def _parse_cmux_json(raw: str) -> Any:
    """Parse the stdout of a ``cmux --json`` call; returns parsed JSON or the raw string on failure."""
    raw = raw.strip()
    try:
        data: object = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return raw
    

class CmuxCli:
    """Low-level subprocess wrapper for the ``cmux`` CLI (ping, tree, IPC commands)."""

    @classmethod
    def get_cmux_bin(cls) -> str:
        """Return the absolute path to the ``cmux`` binary; raises RuntimeError if not found."""
        if not hasattr(cls, "_cmux_bin"):
            import shutil
            cmux_bin = shutil.which("cmux")
            if not cmux_bin:
                raise RuntimeError("cmux not found")
            cls._cmux_bin = cmux_bin
        return cls._cmux_bin

    @classmethod
    def cmux(cls , *args: str , timeout: float | None = None , capture_output = True , text = True ,
             stdin = subprocess.DEVNULL):
        """Run ``cmux <args>`` and return the ``CompletedProcess``; raises RuntimeError on non-zero exit."""
        # cmux = cls.get_cmux_bin()
        # to_shell_string(['cmux', *args])
        r = subprocess.run(
            ['cmux', *args],# [cmux_bin, *args],
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            stdin=stdin,
        )
        if r.returncode != 0:
            raise RuntimeError(f"cmux failed: {r}")
        return r
        
    @classmethod
    def cmux_json(cls , *args: str , timeout: float | None = None , capture_output = True , text = True ,
                  stdin = subprocess.DEVNULL):
        """Run ``cmux --json <args>`` and return the parsed result (dict/list/str)."""
        r = cls.cmux("--json", *args, timeout = timeout, capture_output = capture_output, text = text, stdin = stdin)
        if r.returncode != 0:
            raise RuntimeError(f"cmux json failed: {r.stderr.strip()}")
        return _parse_cmux_json(r.stdout)

    @classmethod
    def cmux_ping(cls) -> tuple[bool, str]:
        """Ping the cmux daemon. Returns (True, detail) on success; (False, error_msg) on failure."""
        try:
            r = cls.cmux("ping" , timeout=5.0)
            detail = r.stderr or r.stdout or ""
            return r.returncode == 0, detail
        except (subprocess.TimeoutExpired, OSError , RuntimeError) as e:
            return False, str(e)
