import pandas as pd
import numpy as np

from src.data import DATAVENDOR
from .portfolio import Portfolio , Port
from .benchmark import Benchmark

class Universe:
    """
    Universe for factor model portfolio
    Parameters:
        name : str
        -all : all stocks
        -top<num> : top <num> stocks
        -top-<num> : exclude bottom <num> stocks
        -benchmark : benchmark
        -<benchmark1>+<benchmark2>+... : combination of benchmarks
    """

    _cache_name : str | None = None
    _cache_portfolio : Portfolio | None = None

    def __init__(self , name : str):
        self.name = name

    def __repr__(self):
        return f'Universe({self.name})'

    @classmethod
    def get_cache_portfolio(cls , name : str) -> Portfolio:
        if cls._cache_name == name and cls._cache_portfolio is not None:
            port = cls._cache_portfolio
        else:
            port = Portfolio(name)
        return port

    @classmethod
    def set_cache_portfolio(cls , portfolio : Portfolio):
        cls._cache_portfolio = portfolio
        cls._cache_name = portfolio.name
    
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

            pf = pf.filter_secid(st_list , exclude = True , inplace = True).filter_secid(small_cp , exclude = True , inplace = True)

        return pf
    
    def get_port(self , date : int , safety = True , exclude_bse = True) -> Port:
        return self.get(date , safety , exclude_bse).get(date)
    
    def to_portfolio(self , dates : list[int] | np.ndarray | None = None) -> Portfolio:
        if dates is None:
            dates = []
        port = self.get_cache_portfolio(self.name)
        for date in np.setdiff1d(dates, port.port_date):
            port.append(self.get_port(date) , ignore_name = True)
        self.set_cache_portfolio(port)
        return port

