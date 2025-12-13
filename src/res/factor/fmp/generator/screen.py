import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any

from ...util import Port , PortCreateResult , PortCreator

from src.basic import DB
from src.data import DATAVENDOR

DEFAULT_SCREEN_RATIO = 0.5
DEFAULT_SORTING_ALPHA = ('pred' , 'gru_day_V1' , None) # (db_src , db_key , column_name)

@dataclass(slots = True)
class ScreeningPortfolioCreatorConfig:
    '''
    Config for Screening Portfolio Generator
    kwargs:
        screen_ratio    : float = DEFAULT_SCREEN_RATIO , ratio of stocks to be screened (used as pool of top stocks)
        sorting_alpha   : tuple[str , str , str | None] = (db_src , db_key , column_name) , source and key of alpha to be used for sorting
    '''
    screen_ratio    : float = DEFAULT_SCREEN_RATIO
    sorting_alpha   : tuple[str , str , str | None] | tuple[str , str] = DEFAULT_SORTING_ALPHA
    n_best          : int = 50
    turn_control    : float = 0.1
    buffer_zone     : float = 0.8
    no_zone         : float = 0.5
    indus_control   : float = 0.1
    sorting_alpha_dates : np.ndarray | Any = None

    def __post_init__(self):
        if self.sorting_alpha_dates is None:
            self.sorting_alpha_dates = DB.dates(*self.sa_src_key)

    @classmethod
    def init_from(cls , print_info : bool = False , **kwargs):
        use_kwargs = {k: v for k, v in kwargs.items() if k in cls.__slots__ and v is not None}
        drop_kwargs = {k: v for k, v in kwargs.items() if k not in cls.__slots__}
        if print_info:
            if use_kwargs : 
                print(f'  --> In initializing {cls.__name__}, used kwargs: {use_kwargs}')
            if drop_kwargs: 
                print(f'  --> In initializing {cls.__name__}, dropped kwargs: {drop_kwargs}')
        return cls(**use_kwargs)

    def get_sorting_alpha(self , model_date : int , indus = True) -> pd.DataFrame:
        if model_date not in self.sorting_alpha_dates:
            if model_date < self.sorting_alpha_dates[0]:
                return pd.DataFrame()
            else:
                date = self.sorting_alpha_dates[self.sorting_alpha_dates < model_date].max()
        else:
            date = model_date
        df = DB.load(*self.sa_src_key , date)
        df = self.sa_rename(df)
        if indus:
            df = DATAVENDOR.INFO.add_indus(df , date , 'unknown')
        return df

    @property
    def stay_num(self) -> float:
        return (1 - self.turn_control) * self.n_best
    
    @property
    def indus_slots(self) -> float:
        return self.n_best * self.indus_control

    @property
    def sa_src_key(self) -> tuple[str , str]:
       return self.sorting_alpha[:2]

    @property
    def sa_col_name(self) -> str:
        if len(self.sorting_alpha) == 3 and self.sorting_alpha[2] is not None:
            return self.sorting_alpha[2]
        else:
            return self.sorting_alpha[1]

    def sa_rename(self , df : pd.DataFrame) -> pd.DataFrame:
        if 'alpha' not in df.columns:
            df = df.rename(columns={self.sa_col_name : 'alpha'})
        if 'alpha' not in df.columns:
            if len(remcols := [col for col in df.columns if col not in ['secid' , 'alpha' , 'index' , 'indus']]) == 1:
                df = df.rename(columns={remcols[0] : 'alpha'})
            else:
                raise ValueError(f'alpha column not found in {df.columns}')
        return df
    
class ScreeningPortfolioCreator(PortCreator):
    DEFAULT_SCREEN_RATIO = DEFAULT_SCREEN_RATIO
    DEFAULT_SORTING_ALPHA = DEFAULT_SORTING_ALPHA

    def __init__(self , name : str):
        super().__init__(name)

    def setup(self , print_info : bool = False , **kwargs):
        self.conf = ScreeningPortfolioCreatorConfig.init_from(print_info = print_info , **kwargs)
        return self
    
    def parse_input(self):
        return self

    def solve(self):
        amodel = self.alpha_model.get_model(self.model_date)
        assert amodel , f'alpha_model is not Amodel at {self.model_date}'
        screening_pool = amodel.to_dataframe(indus=True , na_indus_as = 'unknown')
        if not self.bench_port.is_emtpy(): 
            screening_pool = screening_pool.query('secid in @self.bench_port.secid').copy()
        screening_pool.loc[:, 'rankpct'] = screening_pool['alpha'].rank(pct = True , method = 'first' , ascending = True).fillna(0)
        screening_pool = screening_pool.query('rankpct >= @self.conf.screen_ratio')
        
        pool = self.conf.get_sorting_alpha(self.model_date)
        if pool.empty:
            pool = screening_pool.copy()
        else:
            pool = pool.query('secid in @screening_pool.secid').copy()
        
        pool.loc[:, 'ind_rank']  = pool.groupby('indus')['alpha'].rank(method = 'first' , ascending = False)
        pool.loc[:, 'rankpct']   = pool['alpha'].rank(pct = True , method = 'first' , ascending = True)
        
        pool = pool.merge(self.init_port.port , on = 'secid' , how = 'left').sort_values('alpha' , ascending = False)
        pool.loc[:, 'selected'] = pool['weight'].fillna(0) > 0
        pool.loc[:, 'buffered'] = (
            (pool['rankpct'] >= self.conf.buffer_zone) + (pool['selected'].cumsum() <= self.conf.stay_num) *
             (pool['rankpct'] >= self.conf.no_zone))

        stay = pool.query('selected & buffered')
        stay_secid = stay['secid'].to_numpy() # noqa

        stay_ind_count : pd.Series | Any = stay.groupby('indus')['secid'].count()
        stay_ind_count = stay_ind_count.rename('count')
        
        new_pool = pool.query('secid not in @stay_secid').merge(stay_ind_count , on = 'indus' , how = 'left')
        max_ind_rank = self.conf.indus_slots - new_pool['count'].fillna(0) # noqa
        entry = new_pool.query('ind_rank < @max_ind_rank').sort_values('alpha' , ascending = False).head(self.conf.n_best - stay.shape[0])

        df = pd.concat([stay[['secid' , 'indus']] , entry[['secid' , 'indus']]]).\
            assign(weight = 1).filter(items=['secid' , 'weight']).drop_duplicates(subset=['secid'])
        port = Port(df , date = self.model_date , name = self.name , value = self.value).rescale()
        self.create_result = PortCreateResult(port , len(df) == self.conf.n_best)

        return self

    def output(self):
        if self.detail_infos: 
            self.create_result.analyze(self.bench_port , self.init_port)
        return self