import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

from src.proj import PATH , Logger , Proj , CALENDAR , Dates
from src.res.trading.util.trade_suggestion import TradeSuggestion

from .trading_port import TrackingPort

class TrackingPortfolioManager:
    @classmethod
    def update(cls , reset_ports : list[str] | None = None , indent : int = 0 , vb_level : Any = 1):
        vb_level = Proj.vb(vb_level)
        Logger.note(f'Update: {cls.__name__} since last update!' , indent = indent)
        reset_ports = reset_ports or []
        date = CALENDAR.updated()
        
        assert not reset_ports or all([port in TrackingPort.candidate_ports for port in reset_ports]) , \
            f'expect all reset ports in port_list , got {reset_ports}'
            
        updated_ports = {name:TrackingPort.load(name).build(date , name in reset_ports , indent = indent + 1 , vb_level = vb_level + 2) 
                         for name in TrackingPort.candidate_ports}
        updated_ports = {name:tp for name,tp in updated_ports.items() if not tp.new_ports[date].empty}

        new_ports = {name:tp.new_ports[date] for name,tp in updated_ports.items()}
        last_ports = {name:tp.get_last_port(date).to_dataframe() for name,tp in updated_ports.items()}
            
        if len(updated_ports) == 0: 
            Logger.alert1(f'No Tracking Portfolios Updated at {Dates(date)}' , indent = indent + 1)
        else:
            Logger.success(f'{len(updated_ports)} Tracking Portfolios Updated at {Dates(date)}: [{", ".join(new_ports.keys())}]' , indent = indent + 1 , vb_level = vb_level)
            for port_name in updated_ports:
                in_secids = np.setdiff1d(new_ports[port_name]['secid'], last_ports[port_name]['secid'])
                out_secids = np.setdiff1d(last_ports[port_name]['secid'], new_ports[port_name]['secid'])
                message = f'Port {port_name} : total {len(new_ports[port_name])} , in {len(in_secids)} , out {len(out_secids)}'
                Logger.stdout(message , indent = indent + 2 , vb_level = vb_level + 1)
                if port_name in Proj.Conf.TradingPort.focused_ports:
                    Logger.conclude(message)
                    for suggestion in TradeSuggestion.generate(in_secids, date , 'buy'):
                        Logger.conclude(suggestion.format())
                    for suggestion in TradeSuggestion.generate(out_secids, date , 'sell'):
                        Logger.conclude(suggestion.format())
                    
            path = cls.attachment_path(date)
            pd.concat([df for df in new_ports.values()]).to_csv(path)
            Proj.email_attachments.append(path)

        for name in TrackingPort.candidate_ports:
            tp = TrackingPort.load(name)
            tp.analyze(key_fig = '' , indent = indent + 1 , vb_level = vb_level + 2)

        Logger.success(f'{len(updated_ports)} Tracking Portfolios Analyzed at {Dates(date)}' , indent = indent + 1 , vb_level = vb_level)
                    
    @classmethod
    def attachment_path(cls , date : int) -> Path:
        path = PATH.rslt_trade.joinpath('trading_ports' , f'trading_ports.{date}.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = TrackingPort.load(port_name)
        tp.analyze(start = start , end = end , **kwargs)
        return tp