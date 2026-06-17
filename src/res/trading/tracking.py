"""
Tracking portfolio manager class for the project.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from pathlib import Path

from src.proj import PATH , Proj , CALENDAR , Const , Base , Dates
from src.res.trading.util.trade_suggestion import TradeSuggestion

from .trading_port import TrackingPort

__all__ = ['TrackingPortfolioManager']

class TrackingPortfolioManager(Base.BoundLogger):
    """
    Tracking portfolio manager class for the project.
    """
    @classmethod
    def update(cls , reset_ports : Base.alias.NamesType = None , indent : int = 0 , vb_level : Base.lit.VerbosityLevel = 1):
        """
        Update the tracking portfolios.
        """
        cls.SetClassVB(vb_level , indent)
        cls.logger.note(f'Update since last update!')
        reset_ports = reset_ports or []
        date = CALENDAR.updated()
        
        reset_ports = Base.ensure_name_list(reset_ports , [])
        assert not reset_ports or all([port in TrackingPort.candidate_ports for port in reset_ports]) , \
            f'expect all reset ports in port_list , got {reset_ports}'
        
        cls.logger.stdout(f'Build Tracking Portfolios at {Dates(date)} start ...' , idt = 1 , vb = 1)
        updated_ports : dict[str, TrackingPort] = {}
        for name in TrackingPort.candidate_ports:
            tp = TrackingPort.load(name , vb_level = cls.vb_level + 1, indent = cls.indent + 1)
            with tp.logger.subprocess(idt = 1):
                tp.build(date , name in reset_ports)
                if not tp.new_ports[date].empty:
                    updated_ports[name] = tp

        
        if len(updated_ports) == 0: 
            cls.logger.skipping(f'No Tracking Portfolios Updated at {Dates(date)}' , idt = 1)
        else:
            new_ports = {name:tp.new_ports[date] for name,tp in updated_ports.items()}
            last_ports = {name:tp.get_last_port(date).to_dataframe() for name,tp in updated_ports.items()}
            cls.logger.success(f'{len(updated_ports)} Tracking Portfolios Updated at {Dates(date)}: [{", ".join(new_ports.keys())}]' , idt = 1 , vb = 2)
            for port_name in updated_ports:
                in_secids = np.setdiff1d(new_ports[port_name]['secid'], last_ports[port_name]['secid'])
                out_secids = np.setdiff1d(last_ports[port_name]['secid'], new_ports[port_name]['secid'])
                message = f'Port {port_name} : total {len(new_ports[port_name])} , in {len(in_secids)} , out {len(out_secids)}'
                cls.logger.stdout(message , idt = 2 , vb = 1)
                if port_name in Const.TradingPort.focused_ports:
                    cls.logger.conclude(message)
                    for suggestion in TradeSuggestion.generate(in_secids, date , 'buy'):
                        cls.logger.conclude(suggestion.format())
                    for suggestion in TradeSuggestion.generate(out_secids, date , 'sell'):
                        cls.logger.conclude(suggestion.format())
                    
            path = cls.attachment_path(date)
            pd.concat([df for df in new_ports.values()]).to_csv(path)
            Proj.email_attachments.append(path)

        cls.logger.stdout(f'Analyze Tracking Portfolios at {Dates(date)} start ...' , idt = 1 , vb = 1)
        
        for name in TrackingPort.candidate_ports:
            tp = TrackingPort.load(name , vb_level = cls.vb_level + 1 , indent = cls.indent + 1)
            with tp.logger.subprocess(idt = 1):
                tp.analyze(key_fig = '')
        cls.logger.success(f'{len(TrackingPort.candidate_ports)} Tracking Portfolios Analyzed at {Dates(date)}' , idt = 1)
                    
    @classmethod
    def attachment_path(cls , date : int) -> Path:
        """
        Attachment path for the tracking portfolios.
        """
        path = PATH.rslt_trade.joinpath('trading_ports' , f'trading_ports.{date}.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        """
        Analyze a tracking port.
        """
        tp = TrackingPort.load(port_name)
        tp.analyze(start = start , end = end , **kwargs)
        return tp