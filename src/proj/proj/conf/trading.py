from src.proj.env import MACHINE

__all__ = ['TradingPortConfig' , 'TradingPort']

_trading_port_settings = MACHINE.configs('setting' , 'trading_port')

class TradingPortConfig:
    @property
    def focused_ports(self) -> list[str]:
        """focused ports"""
        return _trading_port_settings['focused_ports']
    @property
    def portfolio_dict(self) -> dict[str , dict]:
        """portfolio dictionary"""
        return _trading_port_settings['portfolio_dict']

TradingPort = TradingPortConfig()
