"""Whether ``gnome-terminal`` is on ``PATH``."""

from __future__ import annotations

import shutil


class GnomeTerminalVerifier:
    @classmethod
    def available(cls) -> bool:
        if not hasattr(cls, "_available"):
            cls._available = bool(shutil.which("gnome-terminal"))
        return cls._available