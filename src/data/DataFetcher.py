import numpy as np
import pandas as pd
import os , re

from dataclasses import dataclass , field
from functools import reduce 
from typing import Callable , ClassVar , List , Literal , Any

from ..environ import DIR
from .DataFetcher_R import DataFetcher_R as RFetcher

# %%
# def stime(compact = False): return time.strftime('%y%m%d%H%M%S' if compact else '%Y-%m-%d %H:%M:%S',time.localtime())
   
@dataclass
class DataFetcher:
    db_src      : str
    db_key      : str
    args        : list = field(default_factory=list)
    fetcher     : Callable | str = 'default'
    
    DB_by_name  : ClassVar[List[str]] = ['information']
    DB_by_date  : ClassVar[List[str]] = ['models' , 'trade' , 'labels']  
    save_option : ClassVar[Literal['feather' , 'parquet']] = 'feather'

    def __post_init__(self):
        if self.fetcher == 'default':
            self.fetcher = self.default_fetcher(self.db_src , self.db_key)

    def __call__(self , date = None , **kwargs) -> Any:
        return self.eval(date , **kwargs) , self.target_path(date)
    
    @classmethod
    def default_fetcher(cls , db_src , db_key):
        if db_src == 'information': return RFetcher.basic_info
        elif db_src == 'models':
            if db_key == 'risk_exp': return RFetcher.risk_model
            elif db_key == 'longcl_exp': return RFetcher.alpha_longcl
        elif db_src == 'trade':
            if db_key == 'day': return RFetcher.trade_day
            elif db_key == 'min': return RFetcher.trade_min
            elif re.match(r'^\d+day$' , db_key): return RFetcher.trade_Xday
            elif re.match(r'^\d+min$' , db_key): return RFetcher.trade_Xmin
        elif db_src == 'labels': return RFetcher.labels
        raise Exception('Unknown default_fetcher')

    def eval(self , date = None , **kwargs) -> Any:
        assert callable(self.fetcher)
        if self.db_src in self.DB_by_name:
            v = self.fetcher(self.db_key , *self.args , **kwargs)
        elif self.db_src in self.DB_by_date:
            v = self.fetcher(date , *self.args , **kwargs)  
        return v
    
    def target_path(self , date = None):
        return self.get_target_path(self.db_src , self.db_key , date , makedir=True)
    
    def source_dates(self):
        return self.get_source_dates(self.db_src , self.db_key)
    
    def target_dates(self):
        return self.get_target_dates(self.db_src , self.db_key)
    
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
    
    @classmethod
    def get_target_path(cls , db_src , db_key , date = None , makedir = False , 
                        force_type : Literal['name' , 'date'] | None = None):
        if db_src in cls.DB_by_name or force_type == 'name':
            db_path = os.path.join(DIR.db , f'DB_{db_src}')
            db_base = f'{db_key}.{cls.save_option}'
        elif db_src in cls.DB_by_date or force_type == 'date':
            assert date is not None
            year_group = int(date) // 10000
            db_path = os.path.join(DIR.db , f'DB_{db_src}' , db_key , str(year_group))
            db_base = f'{db_key}.{str(date)}.{cls.save_option}'
        else:
            raise KeyError(db_src)
        if makedir: os.makedirs(db_path , exist_ok=True)
        return os.path.join(db_path , db_base)
    
    @classmethod
    def get_source_dates(cls , db_src , db_key):
        assert db_src in cls.DB_by_date
        return RFetcher.source_dates('/'.join([db_src , re.sub(r'\d+', '', db_key)]))
    
    @classmethod
    def get_target_dates(cls , db_src , db_key):
        db_path = os.path.join(DIR.db , f'DB_{db_src}' , db_key)
        target_files = RFetcher.list_files(db_path , recur=True)
        target_dates = RFetcher.path_date(target_files)
        return np.array(sorted(target_dates) , dtype=int)
    
    @classmethod
    def load_target_file(cls , db_src , db_key , date = None):
        target_path = cls.get_target_path(db_src , db_key , date)
        if os.path.exists(target_path):
            return cls.load_df(target_path)
        else:
            return None
        
    @classmethod
    def save_df(cls , df , target_path):
        if cls.save_option == 'feather':
            df.to_feather(target_path)
        else:
            df.to_parquet(target_path , engine='fastparquet')

    @classmethod
    def load_df(cls , target_path):
        if cls.save_option == 'feather':
            return pd.read_feather(target_path)
        else:
            return pd.read_parquet(target_path , engine='fastparquet')
