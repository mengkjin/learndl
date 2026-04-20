"""Preferences for the project"""
from __future__ import annotations

from typing import Any
from src.proj.env import MACHINE

class Preference:
    """Preferences for anything in configs/preference/"""
    @property
    def project(self) -> dict[str , Any]:
        """Get the project preference"""
        return MACHINE.config.get('constant/preference/project')

    @property
    def interactive(self) -> dict[str , Any]:
        """Get the interactive preference"""
        return MACHINE.config.get('constant/preference/interactive')

    @property
    def logger(self) -> dict[str , Any]:
        """Get the logger preference"""
        return MACHINE.config.get('constant/preference/logger')

    @property
    def shell_opener(self) -> dict[str , Any]:
        """Get the shell opener preference"""
        return MACHINE.config.get('constant/preference/shell_opener')