import numpy as np
import re

from dataclasses import dataclass , field
from typing import Any , Callable

from .common import DB_BY_DATE , DB_BY_NAME , get_target_path , get_source_dates , get_target_dates
from .fetcher_R import RFetcher

# %%
# def stime(compact = False): return time.strftime('%y%m%d%H%M%S' if compact else '%Y-%m-%d %H:%M:%S',time.localtime())
   
@dataclass
class DataFetcher:
    db_src      : str
    db_key      : str
    args        : list = field(default_factory=list)
    fetcher     : Callable | str = 'default'

    def __post_init__(self):
        if self.fetcher == 'default':
            self.fetcher = self.default_fetcher(self.db_src , self.db_key)

    def __call__(self , date = None , **kwargs) -> Any:
        return self.eval(date , **kwargs) , self.target_path(date)
    
    @classmethod
    def default_fetcher(cls , db_src , db_key):
        if db_src == 'information': return RFetcher.basic_info
        elif db_src == 'models':
            if db_key == 'risk_exp': return RFetcher.risk_exp
            elif db_key == 'risk_cov': return RFetcher.risk_cov
            elif db_key == 'risk_spec': return RFetcher.risk_spec
            elif db_key == 'longcl_exp': return RFetcher.alpha_longcl
        elif db_src == 'trade':
            if db_key == 'day': return RFetcher.trade_day
            elif db_key == 'min': return RFetcher.trade_min
            elif re.match(r'^\d+day$' , db_key): return RFetcher.trade_Xday
            elif re.match(r'^\d+min$' , db_key): return RFetcher.trade_Xmin
        elif db_src == 'labels': return RFetcher.labels
        elif db_src == 'benchmark': return RFetcher.benchmark
        raise Exception('Unknown default_fetcher')

    def eval(self , date = None , **kwargs) -> Any:
        assert callable(self.fetcher)
        if self.db_src in DB_BY_NAME:
            v = self.fetcher(self.db_key , *self.args , **kwargs)
        elif self.db_src in DB_BY_DATE:
            v = self.fetcher(date , *self.args , **kwargs)  
        return v
    
    def target_path(self , date = None):
        return get_target_path(self.db_src , self.db_key , date , makedir=True)
    
    def source_dates(self):
        return get_source_dates(self.db_src , self.db_key)
    
    def target_dates(self):
        return get_target_dates(self.db_src , self.db_key)
    
    def get_update_dates(self , start_dt = None , end_dt = None , trace = 0 , incremental = True , force = False):
        source_dates = self.source_dates()
        target_dates = self.target_dates()
        if force:
            if start_dt is None or end_dt is None:
                raise ValueError(f'start_dt and end_dt must not be None with force update!')
            target_dates = []
        if incremental: 
            if len(target_dates):
                source_dates = source_dates[source_dates >= min(target_dates)]
        if trace > 0 and len(target_dates) > 0: target_dates = target_dates[:-trace]

        new_dates = np.setdiff1d(source_dates , target_dates)
        if start_dt is not None: new_dates = new_dates[new_dates >= start_dt]
        if end_dt   is not None: new_dates = new_dates[new_dates <= end_dt  ]

        return new_dates   