from typing import Any

from src.proj import CALENDAR , Logger , Proj , Dates
from .trading_port import BacktestPort

class BacktestPortfolioManager:
    @classmethod
    def available_ports(cls) -> list[str]:
        return list(BacktestPort.candidate_ports.keys())

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = BacktestPort.load(port_name)
        date = end if end is not None else CALENDAR.updated()
        return tp.build(date).analyze(start = start , end = end , **kwargs)

    @classmethod
    def rebuild(cls , port_name : str , date : int | None = None , export = True , indent : int = 1 , vb_level : Any = 2):
        tp = BacktestPort.load(port_name)
        date = date if date is not None else CALENDAR.updated()
        tp.rebuild(date , export = export , indent = indent , vb_level = vb_level)
        tp.analyze(end = date)
        return tp

    @classmethod
    def update(cls , reset_ports : list[str] | None = None , indent : int = 0 , vb_level : Any = 1):
        vb_level = Proj.vb(vb_level)
        Logger.note(f'Update: {cls.__name__} since last update!' , indent = indent , vb_level = vb_level)
        reset_ports = reset_ports or []
        date = CALENDAR.updated()
        assert not reset_ports or all([port in BacktestPort.candidate_ports for port in reset_ports]) , \
            f'expect all reset ports in port_list , got {reset_ports}'
        updated_ports = {name:BacktestPort.load(name).build(date , indent = indent + 1 , vb_level = 'max') 
                         for name in BacktestPort.candidate_ports}
        updated_ports = {name:tp for name,tp in updated_ports.items() if not tp.new_ports[date].empty}
            
        if len(updated_ports) == 0: 
            Logger.alert1(f'No Backtest Portfolios Updated at {Dates(date)}' , indent = indent + 1)
        else:
            Logger.success(f'{len(updated_ports)} Backtest Portfolios Updated at {Dates(date)}' , indent = indent + 1 , vb_level = vb_level)

        for name in BacktestPort.candidate_ports:
            tp = BacktestPort.load(name)
            tp.analyze(key_fig = '' , indent = indent + 1 , vb_level = vb_level + 2)
        Logger.success(f'{len(updated_ports)} Backtest Portfolios Analyzed at {Dates(date)}' , indent = indent + 1 , vb_level = vb_level)