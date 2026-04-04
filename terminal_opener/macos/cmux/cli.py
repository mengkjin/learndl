"""cmux CLI presence and socket reachability."""

from __future__ import annotations

import subprocess
import json

from typing import Any

__all__ = ["CmuxCli"]

def _parse_cmux_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        data: object = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return raw
    

class CmuxCli:
    @classmethod
    def get_cmux_bin(cls) -> str:
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
        cmux_bin = cls.get_cmux_bin()
        r = subprocess.run(
            [cmux_bin, *args],
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            stdin=stdin,
        )
        if r.returncode != 0:
            raise RuntimeError(f"cmux failed: {r.stderr.strip()}")
        return r
        
    @classmethod
    def cmux_json(cls , *args: str , timeout: float | None = None , capture_output = True , text = True , 
                  stdin = subprocess.DEVNULL):
        r = cls.cmux("--json", *args, timeout = timeout, capture_output = capture_output, text = text, stdin = stdin)
        if r.returncode != 0:
            raise RuntimeError(f"cmux json failed: {r.stderr.strip()}")
        return _parse_cmux_json(r.stdout)

    @classmethod
    def cmux_ping(cls) -> tuple[bool, str]:
        try:
            r = cls.cmux("ping" , timeout=5.0)
            detail = r.stderr or r.stdout or ""
            return r.returncode == 0, detail
        except (subprocess.TimeoutExpired, OSError , RuntimeError) as e:
            return False, str(e)
