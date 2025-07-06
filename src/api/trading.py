from src.basic import Logger
from src.trading import TradingPortfolioTracker

class TradingAPI:
    @staticmethod
    def update(reset_ports : list[str] | None = None): 
        reset_ports = reset_ports or []
        '''
        Update trading portfolios for both laptop and server:
        '''
        with Logger.EnclosedMessage(' update trading portfolios '):
            TradingPortfolioTracker.update(reset_ports = reset_ports)

    @staticmethod
    def Analyze(port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        return TradingPortfolioTracker.analyze(port_name , start , end , **kwargs)
