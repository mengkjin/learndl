import pandas as pd

from typing import Any

from src.proj import Logger , CONST
from ...util import Port , AlphaModel , AlphaScreener , Amodel

class BasicCreatorConfig:
    '''
    Config for Generator of Basic Portfolio
    '''
    slots = ['n_best' , 'turn_control' , 'buffer_zone' , 'no_zone' , 'indus_control']

    def __init__(
        self , 
        n_best : int = CONST.Conf.Fmp.default['top']['n_best'] , 
        turn_control : float = CONST.Conf.Fmp.default['top']['turn_control'] , 
        buffer_zone : float = CONST.Conf.Fmp.default['top']['buffer_zone'] , 
        no_zone : float = CONST.Conf.Fmp.default['top']['no_zone'] , 
        indus_control : float = CONST.Conf.Fmp.default['top']['indus_control'] , 
        **kwargs
    ):
        self.n_best = n_best
        self.turn_control = turn_control
        self.buffer_zone = buffer_zone
        self.no_zone = no_zone
        self.indus_control = indus_control
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join([f"{k}={getattr(self, k)}" for k in self.slots])})'

    @classmethod
    def init_from(cls , indent : int = 1 , vb_level : Any = 3 , **kwargs):
        use_kwargs = {k: v for k, v in kwargs.items() if k in cls.slots and v is not None}
        drop_kwargs = {k: v for k, v in kwargs.items() if k not in cls.slots}
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

    @property
    def stay_num(self) -> float:
        return (1 - self.turn_control) * self.n_best
    
    @property
    def indus_slots(self) -> float:
        return self.n_best * self.indus_control

def generate_top_stock_port(
    model_date : int , 
    sorting_alpha : AlphaModel | Amodel , 
    config : BasicCreatorConfig,
    init_port : Port | None = None, 
    bench_port : Port | None = None, 
    screen_alpha : AlphaScreener | None = None
) -> pd.DataFrame:

    sort_model = sorting_alpha.get_model(model_date)
    assert sort_model , f'sorting_alpha is not Amodel at {model_date} , {sort_model}'
    if init_port is None:
        init_port = Port.none_port(model_date)
    if bench_port is None:
        bench_port = Port.none_port(model_date)

    pool = sort_model.to_dataframe(indus=True , na_indus_as = 'unknown')
    if screen_alpha is not None:
        screened_pool = screen_alpha.screened_pool(model_date , secid = bench_port.secid)
        if screened_pool is not None and len(screened_pool) > 0:
            pool = pool.query('secid in @screened_pool').copy()
    
    pool.loc[:, 'ind_rank']  = pool.groupby('indus')['alpha'].rank(method = 'first' , ascending = False)
    pool.loc[:, 'rankpct']   = pool['alpha'].rank(pct = True , method = 'first' , ascending = True)
    
    pool = pool.merge(init_port.port , on = 'secid' , how = 'left').sort_values('alpha' , ascending = False)
    pool.loc[:, 'selected'] = pool['weight'].astype(float).fillna(0) > 0
    pool.loc[:, 'buffered'] = (
        (pool['rankpct'] >= config.buffer_zone) + (pool['selected'].cumsum() <= config.stay_num) * (pool['rankpct'] >= config.no_zone))

    stay = pool.query('selected & buffered')
    stay_secid = stay['secid'].to_numpy() # noqa

    stay_ind_count : pd.Series | Any = stay.groupby('indus')['secid'].count()
    stay_ind_count = stay_ind_count.rename('count')
    
    new_pool = pool.query('(secid not in @stay_secid)').merge(stay_ind_count , on = 'indus' , how = 'left')
    max_ind_rank = config.indus_slots - new_pool['count'].fillna(0) # noqa
    entry = new_pool.query('ind_rank < @max_ind_rank').sort_values('alpha' , ascending = False).head(config.n_best - stay.shape[0])

    df = pd.concat([stay[['secid' , 'indus']] , entry[['secid' , 'indus']]]).\
        assign(weight = 1).filter(items=['secid' , 'weight']).drop_duplicates(subset=['secid'])
    return df