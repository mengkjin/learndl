"""Linux WezTerm backend: opener, availability verifier, socket discovery, and foreground helpers."""

from .open import WezTermOpener, activate_wezterm
from .verify import WezTermVerifier

__all__ = [
    "WezTermOpener",
    "WezTermVerifier",
    "activate_wezterm",
]
