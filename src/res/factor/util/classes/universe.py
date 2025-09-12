import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.data import DATAVENDOR
from .portfolio import Portfolio , Port
from .benchmark import Benchmark

@dataclass
class Universe:
    name        : str
    
    def get(self , date : int , safety = True , exclude_bse = True) -> Portfolio:
        exchange = ['SZSE','SSE'] if exclude_bse else ['SZSE','SSE','BSE']
        candidates = DATAVENDOR.INFO.get_desc(exchange=exchange).index

        if self.name == 'all':
            pf = Portfolio.from_dataframe(pd.DataFrame({'secid' : candidates , 'date' : date , 'name' : self.name}) , name = self.name)
        elif self.name.startswith('top'):
            top_num = int(self.name.split('.')[0].removeprefix('top'))
            val = DATAVENDOR.TRADE.get_val(DATAVENDOR.TRADE.latest_date('val' , date)).sort_values('circ_mv' , ascending=False)
            val = val.query('secid in @candidates').iloc[:top_num].loc[:,['secid']].\
                reset_index().assign(date = date , name = self.name)
            val['weight'] = 1 / len(val)
            pf = Portfolio.from_dataframe(val , name = self.name)
        elif self.name in Benchmark.AVAILABLES:
            pf = Benchmark(self.name)
        elif '+' in self.name:
            univs = [Benchmark(univ).get(date) for univ in self.name.split('+')]
            pf = Portfolio.from_ports(Port.sum(univs) , name = self.name)
        else:
            raise Exception(f'{self.name} is not a valid benchmark')
        
        assert isinstance(pf , Portfolio) , f'expect Portfolio , got {type(pf)}'

        if safety:
            st_list = DATAVENDOR.INFO.get_st(date)['secid'].to_numpy()
            small_cp = DATAVENDOR.TRADE.get_val(date).query('close < 2.0')['secid'].to_numpy()

            pf = pf.exclude(st_list , True).exclude(small_cp , True)

        return pf
    
    def get_port(self , date : int , safety = True , exclude_bse = True) -> Port:
        return self.get(date , safety , exclude_bse).get(date)
    
    def to_portfolio(self , dates : list[int] | np.ndarray = []) -> Portfolio:
        port = Portfolio(self.name)
        for date in dates:
            port.append(self.get_port(date) , ignore_name = True)
        return port

