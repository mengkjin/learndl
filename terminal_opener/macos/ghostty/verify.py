"""Ghostty.app presence on macOS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class GhosttyVerifier:
    _bundle_id = "com.mitchellh.ghostty"

    @classmethod
    def available(cls) -> bool:
        if not hasattr(cls, "_available"):
            cls._available = cls._ghostty_app_bundle() is not None
        return cls._available

    @classmethod
    def _ghostty_app_bundle(cls) -> str | None:
        if shutil.which("open") is None:
            return None
        for base in (Path("/Applications"), Path.home() / "Applications"):
            candidate = base / "Ghostty.app"
            if candidate.is_dir():
                return str(candidate)
        try:
            r = subprocess.run(
                ["mdfind", f"kMDItemCFBundleIdentifier == '{cls._bundle_id}'"],
                capture_output=True,
                text=True,
                timeout=5.0,
                stdin=subprocess.DEVNULL,
            )
            if r.returncode == 0 and r.stdout.strip():
                line = r.stdout.strip().split("\n", 1)[0].strip()
                if line.endswith(".app"):
                    return line
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None
