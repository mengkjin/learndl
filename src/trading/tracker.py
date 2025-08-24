import pandas as pd

from pathlib import Path

from src.basic import CALENDAR , PATH , Email
from src.trading.util import TradingPort

class TradingPortfolioTracker:
    @classmethod
    def update(cls , reset_ports : list[str] | None = None):
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
            print(f'No trading portfolios updated on {date}')
        else:
            print(f'Trading portfolios updated on {date}: {list(new_ports.keys())}')
            for k in new_ports:
                in_count = (~new_ports[k]['secid'].isin(last_ports[k]['secid'])).sum()
                out_count = (~last_ports[k]['secid'].isin(new_ports[k]['secid'])).sum()
                print(f'    Port {k} : total {len(new_ports[k])} , in {in_count} , out {out_count}')
            path = cls.attachment_path(date)
            pd.concat([df for df in new_ports.values()]).to_csv(path)
            Email.Attach(path)
            path.unlink()

    @classmethod
    def attachment_path(cls , date : int) -> Path:
        path = PATH.rslt_trade.joinpath('daily_trading_ports' , f'trading_ports.{date}.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def analyze(cls , port_name : str , start : int | None = None , end : int | None = None , **kwargs): 
        tp = TradingPort.load(port_name)
        tp.analyze(start = start , end = end , **kwargs)
        return tp