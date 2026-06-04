"""Locate ``wezterm.exe`` via ``PATH`` or common Windows install directories."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _windows_wezterm_exe() -> str | None:
    """Search common Windows install directories (Program Files, LOCALAPPDATA) for ``wezterm.exe``."""
    bases: list[Path] = []
    pf = os.environ.get("ProgramFiles")
    if pf:
        bases.append(Path(pf) / "WezTerm")
    pfx86 = os.environ.get("ProgramFiles(x86)")
    if pfx86:
        bases.append(Path(pfx86) / "WezTerm")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        bases.append(Path(local) / "Programs" / "WezTerm")
    for base in bases:
        candidate = base / "wezterm.exe"
        if candidate.is_file():
            return str(candidate)
    return None


class WezTermVerifier:
    """Locate ``wezterm`` on Windows via PATH or well-known install directories."""

    @classmethod
    def executable(cls) -> str | None:
        """Return the absolute path to the WezTerm executable, or None if not found (result cached)."""
        if not hasattr(cls, "_executable"):
            found = shutil.which("wezterm")
            if not found and sys.platform == "win32":
                found = _windows_wezterm_exe()
            cls._executable = found
        return cls._executable

    @classmethod
    def available(cls) -> bool:
        """Return True if a WezTerm executable can be located."""
        return cls.executable() is not None
