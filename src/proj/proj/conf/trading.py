from src.proj.env import MACHINE

__all__ = ['TradingPortConfig' , 'TradingPort']

_trading_port_settings = MACHINE.configs('setting' , 'trading_port')

class TradingPortConfig:
    @property
    def focused_ports(self) -> list[str]:
        """focused ports"""
        return _trading_port_settings['focused_ports']
    @property
    def tracking_ports(self) -> dict[str , dict]:
        """trading ports"""
        return _trading_port_settings['trading_ports']
    @property
    def backtest_ports(self) -> dict[str , dict]:
        """backtest ports"""
        return _trading_port_settings['backtest_ports']

TradingPort = TradingPortConfig()
