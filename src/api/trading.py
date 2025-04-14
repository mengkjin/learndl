from src.func.display import EnclosedMessage
from src.trading.tracker import TradingPortfolioTracker

class TradingAPI:
    @staticmethod
    def update(reset_ports = []): 
        '''
        Update trading portfolios for both laptop and server:
        '''
        with EnclosedMessage(' update trading portfolios '):
            TradingPortfolioTracker.update(reset_ports = reset_ports)
