from src.res.trading import TradingPortfolioTracker

from .util import wrap_update

class TradingAPI:
    @classmethod
    def update(cls, reset_ports : list[str] | None = None): 
        """
        Update trading portfolios for both laptop and server:
        """
        reset_ports = reset_ports or []
        wrap_update(TradingPortfolioTracker.update , 'update trading portfolios' , reset_ports = reset_ports)

    @staticmethod
    def Analyze(port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        return TradingPortfolioTracker.analyze(port_name , start , end , **kwargs)
