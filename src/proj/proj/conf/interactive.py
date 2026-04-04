"""Streamlit / interactive app metadata (title, version, refresh)."""

from __future__ import annotations

from src.proj.core import singleton
from src.__version__ import __version__

@singleton
class InteractiveConfig:
    """
    Static UI defaults for the interactive front-end:
    - version: version of the app
    - recommended_explorer: recommended explorer of the app
    - page_title: title of the app
    - pending_features: pending features of the app
    - auto_refresh_interval: auto refresh interval of the app
    """
    @property
    def version(self) -> str:
        """version of the app"""
        return __version__

    @property
    def recommended_explorer(self) -> str:
        """recommended explorer of the app"""
        return 'chrome'

    @property
    def page_title(self) -> str:
        """title of the app"""
        return 'Learndl'

    @property
    def pending_features(self) -> list[str]:
        """pending features of the app"""
        return []

    @property
    def auto_refresh_interval(self) -> int:
        """auto refresh interval of the app"""
        return 0
