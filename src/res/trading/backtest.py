from __future__ import annotations
from typing import Any

from src.proj import CALENDAR , Dates , BaseClass
from .trading_port import BacktestPort

class BacktestPortfolioManager(BaseClass.BoundLogger):
    @classmethod
    def available_ports(cls) -> list[str]:
        return list(BacktestPort.candidate_ports.keys())

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = BacktestPort.load(port_name)
        date = end if end is not None else CALENDAR.updated()
        return tp.build(date).analyze(start = start , end = end , **kwargs)

    @classmethod
    def rebuild(cls , port_name : str , date : int | None = None , export = True , analyze = True , indent : int = 1 , vb_level : Any = 2):
        tp = BacktestPort.load(port_name , indent = indent , vb_level = vb_level)
        date = date if date is not None else CALENDAR.updated()
        tp.rebuild(date , export = export)
        if analyze:
            tp.analyze(end = date)
        return tp

    @classmethod
    def update(cls , reset_ports : list[str] | None = None , indent : int = 0 , vb_level : Any = 1):
        cls.SetClassVB(vb_level , indent)
        cls.logger.note(f'{cls.__name__} : Update since last update!')
        reset_ports = reset_ports or []
        date = CALENDAR.updated()
        assert not reset_ports or all([port in BacktestPort.candidate_ports for port in reset_ports]) , \
            f'expect all reset ports in port_list , got {reset_ports}'
        updated_ports = {name:BacktestPort.load(name , indent = indent + 1 , vb_level = 'max').build(date) 
                         for name in BacktestPort.candidate_ports}
        updated_ports = {name:tp for name,tp in updated_ports.items() if not tp.new_ports[date].empty}
            
        if len(updated_ports) == 0: 
            cls.logger.alert1(f'No Backtest Portfolios Updated at {Dates(date)}' , idt = 1)
        else:
            cls.logger.success(f'{len(updated_ports)} Backtest Portfolios Updated at {Dates(date)}' , idt = 1 , vb = 2)

        for name in BacktestPort.candidate_ports:
            tp = BacktestPort.load(name , vb_level = vb_level + 2 , indent = indent + 1)
            tp.analyze(key_fig = '')
        cls.logger.success(f'{len(BacktestPort.candidate_ports)} Backtest Portfolios Analyzed at {Dates(date)}' , idt = 1)