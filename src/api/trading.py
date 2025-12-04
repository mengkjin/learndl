from src.res.trading import TradingPort , TradingPortfolioTracker , TradingPortfolioBacktestor

from src.proj import Logger
from .util import wrap_update

class TradingAPI:
    @classmethod
    def available_ports(cls , backtest : bool | None = None) -> list[str]:
        tradeports = TradingPort.portfolio_dict()
        if backtest is None:
            return list(tradeports.keys())
        elif backtest:
            return [key for key, value in tradeports.items() if value.get('backtest' , False)]
        else:
            return [key for key, value in tradeports.items() if not value.get('backtest' , False)]

    @classmethod
    def update(cls, reset_ports : list[str] | None = None): 
        """
        Update trading portfolios for both laptop and server:
        """
        reset_ports = reset_ports or []
        wrap_update(TradingPortfolioTracker.update , 'update trading portfolios' , reset_ports = reset_ports)

    @classmethod
    def Analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        assert port_name in cls.available_ports() , f'port name {port_name} is not a valid analyze port'
        return TradingPortfolioTracker.analyze(port_name , start , end , **kwargs)

    @classmethod
    def Backtest(cls , port_name_starter : str , start : int | None = None , end : int | None = None , **kwargs): 
        available_ports = cls.available_ports(backtest = True)
        ports = [port for port in available_ports if port.startswith(port_name_starter)]
        if ports:
            Logger.warning(f'multiple backtest ports found for {port_name_starter}: {ports}')
            for port in ports:
                with Logger.ParagraphIII(f'backtest {port}'):
                    TradingPortfolioBacktestor.analyze(port , start , end , **kwargs)
        elif len(ports) == 0:
            Logger.error(f'no backtest ports found starting with {port_name_starter}')
            Logger.info(f'available backtest ports: {available_ports}')
        
