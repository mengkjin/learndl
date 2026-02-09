import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any

from ...util import Port , PortCreateResult , PortCreator

from src.proj import Logger , DB
from src.data import DATAVENDOR

DEFAULT_SCREEN_RATIO = 0.5
DEFAULT_SCREENER = ('pred' , 'gru_day_V1' , None) # (db_src , db_key , column_name)

@dataclass(slots = True)
class ReinforcePortfolioCreatorConfig:
    '''
    Config for Reverse Screening Portfolio Generator
    Screen (by screener) + Sorting (by target alpha) = (RevScreen) = (Reinforce)
    Screening is based on a screen alpha to screen stocks (e.g. choose top 50% of stocks with the gru_day_V1)
    Then, the stocks are sorted by the target alpha (e.g. choose top 50 of pred)

    kwargs:
        screen_ratio    : float = DEFAULT_SCREEN_RATIO , ratio of stocks to be screened (used as pool of top stocks)
        screener   : tuple[str , str , str | None] = (db_src , db_key , column_name) , source and key of alpha to be used for sorting
    '''
    screener        : tuple[str , str , str | None] | tuple[str , str] = DEFAULT_SCREENER
    screen_ratio    : float = DEFAULT_SCREEN_RATIO
    n_best          : int = 50
    turn_control    : float = 0.1
    buffer_zone     : float = 0.8
    no_zone         : float = 0.5
    indus_control   : float = 0.1

    @classmethod
    def init_from(cls , indent : int = 1 , vb_level : int = 3 , **kwargs):
        use_kwargs = {k: v for k, v in kwargs.items() if k in cls.__slots__ and v is not None}
        drop_kwargs = {k: v for k, v in kwargs.items() if k not in cls.__slots__}
        if use_kwargs and drop_kwargs: 
            kwargs_str = f'used kwargs: {use_kwargs}, dropped kwargs: {drop_kwargs}'
        elif use_kwargs:
            kwargs_str = f'used kwargs: {use_kwargs}'
        elif drop_kwargs:
            kwargs_str = f'dropped kwargs: {drop_kwargs}'
        else:
            kwargs_str = 'no kwargs used'
        Logger.stdout(f'{cls.__name__}.init_from: {kwargs_str}' , indent = indent , vb_level = vb_level)
        return cls(**use_kwargs)

    def get_screen_alpha(self , model_date : int , indus = True) -> pd.DataFrame:
        df = DB.load(*self.screener_src_key , model_date , closest = True)
        df = self.screener_rename(df)
        if indus:
            df = DATAVENDOR.INFO.add_indus(df , model_date , 'unknown')
        return df

    @property
    def stay_num(self) -> float:
        return (1 - self.turn_control) * self.n_best
    
    @property
    def indus_slots(self) -> float:
        return self.n_best * self.indus_control

    @property
    def screener_src_key(self) -> tuple[str , str]:
       return self.screener[:2]

    @property
    def screener_col(self) -> str:
        if len(self.screener) == 3 and self.screener[2] is not None:
            return self.screener[2]
        else:
            return self.screener[1]

    @property
    def screener_dates(self) -> np.ndarray:
        return DB.dates(*self.screener_src_key)

    def screener_rename(self , df : pd.DataFrame) -> pd.DataFrame:
        if 'alpha' not in df.columns:
            df = df.rename(columns={self.screener_col : 'alpha'})
        if 'alpha' not in df.columns:
            if len(remcols := [col for col in df.columns if col not in ['secid' , 'alpha' , 'index' , 'indus']]) == 1:
                df = df.rename(columns={remcols[0] : 'alpha'})
            else:
                raise ValueError(f'alpha column not found in {df.columns}')
        return df
    
class ReinforcePortfolioCreator(PortCreator):
    DEFAULT_SCREEN_RATIO = DEFAULT_SCREEN_RATIO
    DEFAULT_SCREEN_ALPHA = DEFAULT_SCREENER

    def setup(self , indent : int = 1 , vb_level : int = 3 , **kwargs):
        self.conf = ReinforcePortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self
    
    def parse_input(self):
        return self

    def solve(self):
        amodel = self.alpha_model.get_model(self.model_date)
        assert amodel , f'alpha_model is not Amodel at {self.model_date}'
        pool = amodel.to_dataframe(indus=True , na_indus_as = 'unknown')

        screening_pool = self.conf.get_screen_alpha(self.model_date , indus = True)
        if not screening_pool.empty:
            if not self.bench_port.is_emtpy(): 
                screening_pool = screening_pool.query('secid in @self.bench_port.secid').copy()
            screening_pool.loc[:, 'rankpct'] = screening_pool['alpha'].rank(pct = True , method = 'first' , ascending = True).fillna(0)
            screening_pool = screening_pool.query('rankpct >= @self.conf.screen_ratio')
        
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