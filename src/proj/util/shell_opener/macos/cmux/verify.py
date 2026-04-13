"""cmux CLI presence and socket reachability."""

from __future__ import annotations

import shutil
__all__ = ["CmuxVerifier"]

class CmuxVerifier:
    """Check whether the ``cmux`` binary is reachable on PATH."""

    @classmethod
    def available(cls) -> bool:
        """Return True if ``cmux`` is found on ``PATH`` (result cached after first call)."""
        if not hasattr(cls, "_available"):
            cls._available = shutil.which("cmux") is not None
        return cls._available