"""Whether ``wezterm`` is on ``PATH``."""

from __future__ import annotations

import shutil


class WezTermVerifier:
    """Check whether the ``wezterm`` binary is reachable on PATH."""

    @classmethod
    def available(cls) -> bool:
        """Return True if ``wezterm`` is found on ``PATH`` (result cached after first call)."""
        if not hasattr(cls, "_available"):
            cls._available = bool(shutil.which("wezterm"))
        return cls._available
