from src.proj import CALENDAR
from .trading_port import BacktestPort

class TradingPortfolioBacktestor:
    @classmethod
    def available_ports(cls) -> list[str]:
        return list(BacktestPort.candidate_ports.keys())

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = BacktestPort.load(port_name)
        date = end if end is not None else CALENDAR.updated()
        return tp.build(date).analyze(start = start , end = end , **kwargs)

    @classmethod
    def rebuild(cls , port_name : str , date : int | None = None , export = True , indent : int = 1 , vb_level : int = 2):
        tp = BacktestPort.load(port_name)
        date = date if date is not None else CALENDAR.updated()
        tp.rebuild(date , export = export , indent = indent , vb_level = vb_level)
        tp.analyze(end = date)
        return tp