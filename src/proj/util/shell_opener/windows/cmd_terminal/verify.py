"""Windows cmd.exe availability."""

from __future__ import annotations

import shutil

class CmdTerminalVerifier:
    """Check whether ``cmd.exe`` is available (always true on a standard Windows install)."""

    @classmethod
    def available(cls) -> bool:
        """Return True if ``cmd.exe`` or ``cmd`` is found on ``PATH`` (result cached after first call)."""
        if not hasattr(cls, "_available"):
            cls._available = shutil.which("cmd.exe") is not None or shutil.which("cmd") is not None
        return cls._available
