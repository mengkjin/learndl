"""Ghostty.app presence on macOS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class GhosttyVerifier:
    """Check whether Ghostty.app is installed (searches standard app directories and Spotlight)."""

    _bundle_id = "com.mitchellh.ghostty"

    @classmethod
    def available(cls) -> bool:
        """Return True if a Ghostty.app bundle is found on this machine (result cached after first call)."""
        if not hasattr(cls, "_available"):
            cls._available = cls._ghostty_app_bundle() is not None
        return cls._available

    @classmethod
    def _ghostty_app_bundle(cls) -> str | None:
        """Return the path to Ghostty.app, or None if not found via standard paths or Spotlight."""
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
