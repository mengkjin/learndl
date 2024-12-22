from src.func.display import EnclosedMessage
from src.trading.builder import TradingPortfolioBuilder

class TradingAPI:
    @staticmethod
    def update(reset_ports = []): 
        '''
        Update trading portfolios for both laptop and server:
        '''
        with EnclosedMessage(' update trading portfolios '):
            TradingPortfolioBuilder.update(reset_ports = reset_ports)
