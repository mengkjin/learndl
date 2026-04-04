"""cmux CLI presence and socket reachability."""

from __future__ import annotations

import shutil
__all__ = ["CmuxVerifier"]

class CmuxVerifier:
    @classmethod
    def available(cls) -> bool:
        if not hasattr(cls, "_available"):
            cls._available = shutil.which("cmux") is not None
        return cls._available