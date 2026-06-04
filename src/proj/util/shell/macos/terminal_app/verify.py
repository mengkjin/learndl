"""Check availability of Terminal.app automation."""

from __future__ import annotations

import shutil


class TerminalAppVerifier:
    """Check whether Terminal.app automation is available via ``osascript``."""

    @classmethod
    def available(cls) -> bool:
        """Return True if ``osascript`` is found on ``PATH`` (result cached after first call)."""
        if not hasattr(cls, "_available"):
            cls._available = shutil.which("osascript") is not None
        return cls._available

