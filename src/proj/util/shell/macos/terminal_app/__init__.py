"""macOS Terminal.app backend: opener and availability verifier."""

from .open import TerminalAppOpener
from .verify import TerminalAppVerifier

__all__ = ["TerminalAppOpener", "TerminalAppVerifier"]
