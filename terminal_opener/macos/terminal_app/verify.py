"""Check availability of Terminal.app automation."""

from __future__ import annotations

import shutil


class TerminalAppVerifier:
    @classmethod
    def available(cls) -> bool:
        if not hasattr(cls, "_available"):
            cls._available = shutil.which("osascript") is not None
        return cls._available

