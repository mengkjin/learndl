"""Windows cmd.exe terminal backend: opener and availability verifier."""

from .open import CmdTerminalOpener
from .verify import CmdTerminalVerifier

__all__ = ["CmdTerminalOpener", "CmdTerminalVerifier"]
