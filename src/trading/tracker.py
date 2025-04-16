import pandas as pd

from pathlib import Path

from src.basic import CALENDAR , PATH , Email
from src.trading.util import TradingPort

class TradingPortfolioTracker:
    @classmethod
    def update(cls , reset_ports : list[str] = []):
        date = CALENDAR.updated()
        ports = TradingPort.portfolio_dict()

        assert not reset_ports or all([port in ports for port in reset_ports]) , \
            f'expect all reset ports in port_list , got {reset_ports}'
            
        updated_ports = {k:TradingPort(k , **v).go_backtest().build_portfolio(date , k in reset_ports)
                         for k,v in ports.items()}
        updated_ports = {k:v for k,v in updated_ports.items() if not v.empty}
            
        if len(updated_ports) == 0: 
            print(f'No trading portfolios updated on {date}')
        else:
            print(f'Trading portfolios updated on {date}: {list(updated_ports.keys())}')
            path = cls.attachment_path(date)
            pd.concat([df for df in updated_ports.values()]).to_csv(path)
            Email.attach(path)


    @classmethod
    def attachment_path(cls , date : int) -> Path:
        path = PATH.log_update.joinpath('trading_ports',f'trading_ports.{date}.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path