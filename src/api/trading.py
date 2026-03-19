from src.res.trading import TrackingPortfolioManager , BacktestPortfolioManager

from src.proj import Logger , Proj
from .util import wrap_update

class TradingAPI:
    @classmethod
    def available_ports(cls , backtest : bool | None = None) -> list[str]:
        if backtest is None:
            return list(Proj.Conf.TradingPort.tracking_ports.keys()) + list(Proj.Conf.TradingPort.backtest_ports.keys())
        elif backtest:
            return list(Proj.Conf.TradingPort.backtest_ports.keys())
        else:
            return list(Proj.Conf.TradingPort.tracking_ports.keys())

    @classmethod
    def backtest_rebuild(cls , port_name : str):
        """
        Rebuild backtest portfolio for a given port:
        """
        BacktestPortfolioManager.rebuild(port_name).analyze()

    @classmethod
    def update(cls): 
        """
        Update trading portfolios for both laptop and server:
        """
        def update_trading_ports():
            TrackingPortfolioManager.update()
            BacktestPortfolioManager.update()
        wrap_update(update_trading_ports , 'update trading portfolios')

    @classmethod
    def Analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs):
        if port_name in cls.available_ports(backtest = True):
            return BacktestPortfolioManager.analyze(port_name , start , end , **kwargs)
        elif port_name in cls.available_ports(backtest = False):
            return TrackingPortfolioManager.analyze(port_name , start , end , **kwargs)
        else:
            raise ValueError(f'port name {port_name} is not a valid analyze port')

    @classmethod
    def Backtest(cls , port_name_starter : str , start : int | None = None , end : int | None = None , **kwargs): 
        available_ports = cls.available_ports(backtest = True)
        ports = [port for port in available_ports if port.startswith(port_name_starter)]
        if ports:
            Logger.stdout(f'multiple backtest ports found for {port_name_starter}: {ports}')
            for port in ports:
                with Logger.Paragraph(f'backtest {port}' , 3):
                    BacktestPortfolioManager.analyze(port , start , end , **kwargs)
        elif len(ports) == 0:
            Logger.error(f'no backtest ports found starting with {port_name_starter}')
            Logger.stdout(f'available backtest ports: {available_ports}')
        
