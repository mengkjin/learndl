from src.proj import CALENDAR
from src.res.trading.util import TradingPort

class TradingPortfolioBacktestor:
    @classmethod
    def available_ports(cls) -> list[str]:
        return [key for key, value in TradingPort.portfolio_dict().items() if value.get('backtest' , False)]

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = TradingPort.load(port_name)
        assert tp.backtest , f'port {port_name} is not a backtest port'
        date = end if end is not None else CALENDAR.updated()
        return tp.build(date).analyze(start = start , end = end , **kwargs)