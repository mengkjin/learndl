"""Trading and backtest portfolio lists from ``setting/trading_port``."""

from __future__ import annotations
from src.proj.env import MACHINE
from src.proj.core import singleton

__all__ = ['TradingPortConfig']

@singleton
class TradingPortConfig:
    """
    Thin view over YAML-loaded trading port metadata:
    - focused_ports [list[str]]: focused ports
    - tracking_ports [dict[str , dict]]: trading ports
    - backtest_ports [dict[str , dict]]: backtest ports
    """
    _settings = MACHINE.configs('setting' , 'trading_port')

    @property
    def focused_ports(self) -> list[str]:
        """focused ports"""
        return self._settings['focused_ports']
    @property
    def tracking_ports(self) -> dict[str , dict]:
        """trading ports"""
        return self._settings['trading_ports']
    @property
    def backtest_ports(self) -> dict[str , dict]:
        """backtest ports"""
        return self._settings['backtest_ports']