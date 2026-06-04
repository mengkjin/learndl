"""macOS WezTerm backend: opener, availability verifier, and foreground-activation helper."""

from .open import WezTermOpener, activate_wezterm
from .verify import WezTermVerifier

__all__ = [
    "WezTermOpener",
    "WezTermVerifier",
    "activate_wezterm",
]
