"""Trading and backtest portfolio lists from ``setting/trading_port``."""

from __future__ import annotations
from src.proj.env import MACHINE
from src.proj.core import singleton

__all__ = ['FMPConfig']

@singleton
class FMPConfig:
    """
    Thin view over YAML-loaded FMP metadata:
    - default: default FMP config
    """
    _settings = MACHINE.configs('util' , 'factor' , 'fmp')

    @property
    def default(self) -> dict[str , dict]:
        """default FMP config"""
        return self._settings['default']