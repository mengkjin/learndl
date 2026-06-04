"""Ghostty terminal backend: opener and availability verifier."""

from .open import GhosttyOpener
from .verify import GhosttyVerifier

__all__ = ["GhosttyOpener", "GhosttyVerifier"]
