import pandas as pd

from pathlib import Path

from src.basic import CALENDAR , PATH , Email
from src.trading.util import TradingPort

class TradingPortfolioBuilder:
    @classmethod
    def update(cls , reset_ports : list[str] = []):
        date = CALENDAR.updated()
        pfs : list[pd.DataFrame] = []
        port_list = PATH.read_yaml(Path('port_list.yaml').absolute())

        if reset_ports:
            ports , reset = reset_ports , True
            assert all([port in port_list for port in reset_ports]) , \
                f'expect all reset ports in port_list , got {reset_ports}'
        else:
            ports , reset = list(port_list.keys()) , False
            
        updated_ports = []
        for port_name in ports:
            port = TradingPort.load(port_name , **port_list[port_name])
            builder = port.get_builder(date , reset)
            pf = builder.build(date).port.to_dataframe()
            if pf.empty: continue

            if port.first_date(date):
                pf['value'] = port.init_value

            path = port.port_path(date)
            path.parent.mkdir(parents=True, exist_ok=True)
            pf.loc[:,['secid' , 'weight' , 'value']].to_csv(path)
            
            pfs.append(pf)
            updated_ports.append(port.name)
            
        if len(updated_ports) == 0: 
            print(f'No trading portfolios updated on {date}')
            return
        else:
            print(f'Trading portfolios updated on {date}: {updated_ports}')
        total_port = pd.concat([df for df in pfs])
        total_path = PATH.log_update.joinpath('trading_ports',f'trading_ports.{date}.csv')
        total_path.parent.mkdir(parents=True, exist_ok=True)
        total_port.to_csv(total_path)
        Email.attach(total_path)