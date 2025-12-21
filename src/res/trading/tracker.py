import numpy as np
import pandas as pd

from pathlib import Path

from src.proj import PATH , Logger , ProjStates
from src.basic import CALENDAR
from src.res.trading.util import TradingPort

FOCUSED_PORTS = ['use_daily']

class TradingPortfolioTracker:
    @classmethod
    def update(cls , reset_ports : list[str] | None = None):
        Logger.info(f'Update: {cls.__name__} since last update!')
        reset_ports = reset_ports or []
        date = CALENDAR.updated()
        ports = TradingPort.portfolio_dict()

        assert not reset_ports or all([port in ports for port in reset_ports]) , \
            f'expect all reset ports in port_list , got {reset_ports}'
            
        updated_ports = {k:TradingPort(k , **v).build(date , k in reset_ports) for k,v in ports.items()}
        updated_ports = {k:v for k,v in updated_ports.items() if not v.new_ports[date].empty}

        new_ports = {k:v.new_ports[date] for k,v in updated_ports.items()}
        last_ports = {k:v.get_last_port(date).to_dataframe() for k,v in updated_ports.items()}
            
        if len(updated_ports) == 0: 
            Logger.stdout(f'No trading portfolios updated on {date}')
        else:
            Logger.stdout(f'Trading portfolios updated on {date}: {list(new_ports.keys())}')
            for port_name in new_ports:
                in_secids = np.setdiff1d(new_ports[port_name]['secid'], last_ports[port_name]['secid'])
                out_secids = np.setdiff1d(last_ports[port_name]['secid'], new_ports[port_name]['secid'])
                message = f'Port {port_name} : total {len(new_ports[port_name])} , in {len(in_secids)} , out {len(out_secids)}'
                Logger.stdout(f'    {message}')
                if port_name in FOCUSED_PORTS:
                    in_detail = f'include new secids: {in_secids}'
                    out_detail = f'exclude old secids: {out_secids}'
                    Logger.conclude(message)
                    Logger.conclude(in_detail)
                    Logger.conclude(out_detail)

            path = cls.attachment_path(date)
            pd.concat([df for df in new_ports.values()]).to_csv(path)
            ProjStates.email_attachments.append(path)

    @classmethod
    def attachment_path(cls , date : int) -> Path:
        path = PATH.rslt_trade.joinpath('trading_ports' , f'trading_ports.{date}.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = TradingPort.load(port_name)
        tp.analyze(start = start , end = end , **kwargs)
        return tp