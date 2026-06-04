"""GNOME Terminal backend: opener and availability verifier."""

from .open import GnomeTerminalOpener
from .verify import GnomeTerminalVerifier

__all__ = [
    "GnomeTerminalOpener",
    "GnomeTerminalVerifier",
]
