"""Windows cmd.exe availability."""

from __future__ import annotations

import shutil

class CmdTerminalVerifier:
    @classmethod
    def available(cls) -> bool:
        if not hasattr(cls, "_available"):
            cls._available = shutil.which("cmd.exe") is not None or shutil.which("cmd") is not None
        return cls._available
