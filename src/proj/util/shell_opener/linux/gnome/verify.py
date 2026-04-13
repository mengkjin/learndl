"""Whether ``gnome-terminal`` is on ``PATH``."""

from __future__ import annotations

import shutil


class GnomeTerminalVerifier:
    """Check whether ``gnome-terminal`` is reachable on PATH."""

    @classmethod
    def available(cls) -> bool:
        """Return True if ``gnome-terminal`` is found on ``PATH`` (result cached after first call)."""
        if not hasattr(cls, "_available"):
            cls._available = bool(shutil.which("gnome-terminal"))
        return cls._available