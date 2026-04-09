"""Whether ``wezterm`` is on ``PATH``."""

from __future__ import annotations

import shutil


class WezTermVerifier:
    @classmethod
    def available(cls) -> bool:
        if not hasattr(cls, "_available"):
            cls._available = bool(shutil.which("wezterm"))
        return cls._available
