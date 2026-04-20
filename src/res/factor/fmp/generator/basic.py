import pandas as pd

from abc import abstractmethod
from typing import Any

from src.proj import Logger , Const
from ...util import Port , AlphaModel , AlphaScreener , Amodel , AlphaComposite , PortCreator , PortCreateResult

class BasicCreatorConfig:
    '''
    Config for Generator of Basic Portfolio
    '''
    slots = ['n_best' , 'turn_control' , 'buffer_zone' , 'no_zone' , 'indus_control' , 'sorter' , 'screener' , 'screen_ratio']

    def __init__(
        self , 
        sorter : str | list[str] | None = None , 
        screener : str | list[str] | None = None , 
        n_best : int = Const.Factor.FMP.creator['generator']['n_best'] , 
        turn_control : float = Const.Factor.FMP.creator['generator']['turn_control'] , 
        buffer_zone : float = Const.Factor.FMP.creator['generator']['buffer_zone'] , 
        no_zone : float = Const.Factor.FMP.creator['generator']['no_zone'] , 
        indus_control : float = Const.Factor.FMP.creator['generator']['indus_control'] , 
        screen_ratio : float = Const.Factor.FMP.creator['generator']['screen_ratio'] , 
        **kwargs
    ):
        self.n_best = n_best
        self.turn_control = turn_control
        self.buffer_zone = buffer_zone
        self.no_zone = no_zone
        self.indus_control = indus_control
        self.kwargs = kwargs
        self.sorter : list[str] = [sorter] if isinstance(sorter , str) else (sorter or ['self'])
        self.screener : list[str] = [screener] if isinstance(screener , str) else (screener or [])
        self.screen_ratio : float = screen_ratio
        assert 'self' in (self.sorter + self.screener) , f'sorter or screener must contain "self" , but got {sorter} and {screener}'
        
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
    def self_in_sorter(self) -> bool:
        return 'self' in self.sorter
    @property
    def self_in_screener(self) -> bool:
        return 'self' in self.screener
    @property
    def alpha_sorter(self) -> AlphaComposite:
        if not hasattr(self , '_sorter_alpha'):
            self._sorter_alpha = AlphaComposite([alpha for alpha in self.sorter if alpha != 'self'])
        return self._sorter_alpha   
    @property
    def alpha_screener(self) -> AlphaScreener:
        if not hasattr(self , '_screener_alpha'):
            self._screener_alpha = AlphaScreener([alpha for alpha in self.screener if alpha != 'self'] , ratio = self.screen_ratio)
        return self._screener_alpha

    @property
    def stay_num(self) -> float:
        return (1 - self.turn_control) * self.n_best
    
    @property
    def indus_slots(self) -> float:
        return self.n_best * self.indus_control

    def get_screened_pool(
        self ,
        model_date : int , 
        alpha_model : AlphaModel | Amodel ,
        bench_port : Port | None = None, 
    ):
        secid = None if bench_port is None else bench_port.secid
        secid = self.alpha_screener.screened_pool(model_date , secid , other_models = alpha_model if self.self_in_screener else None)
        return secid

    def generate_top_stock_port(
        self ,
        model_date : int , 
        alpha_model : AlphaModel | Amodel ,
        init_port : Port | None = None, 
        bench_port : Port | None = None, 
    ) -> pd.DataFrame:
        sorting_alpha = self.alpha_sorter.get(model_date , alpha_model if self.self_in_sorter else None)
        sort_model = sorting_alpha.get_model(model_date)
        if not sort_model:
            sort_model = alpha_model.get_model(model_date)

        screened_pool = self.get_screened_pool(model_date , alpha_model , bench_port)
        if init_port is None:
            init_port = Port.none_port(model_date)

        pool = sort_model.to_dataframe(indus=True , na_indus_as = 'unknown')
        if screened_pool is not None and len(screened_pool) > 0:
            pool = pool.query('secid in @screened_pool').copy()
        
        pool.loc[:, 'ind_rank']  = pool.groupby('indus')['alpha'].rank(method = 'first' , ascending = False)
        pool.loc[:, 'rankpct']   = pool['alpha'].rank(pct = True , method = 'first' , ascending = True)
        
        pool = pool.merge(init_port.port , on = 'secid' , how = 'left').sort_values('alpha' , ascending = False)
        pool.loc[:, 'selected'] = pool['weight'].astype(float).fillna(0) > 0
        pool.loc[:, 'buffered'] = (
            (pool['rankpct'] >= self.buffer_zone) + (pool['selected'].cumsum() <= self.stay_num) * (pool['rankpct'] >= self.no_zone))

        stay = pool.query('selected & buffered')
        stay_secid = stay['secid'].to_numpy() # noqa

        stay_ind_count : pd.Series | Any = stay.groupby('indus')['secid'].count()
        stay_ind_count = stay_ind_count.rename('count')
        
        new_pool = pool.query('(secid not in @stay_secid)').merge(stay_ind_count , on = 'indus' , how = 'left')
        max_ind_rank = self.indus_slots - new_pool['count'].fillna(0) # noqa
        entry = new_pool.query('ind_rank < @max_ind_rank').sort_values('alpha' , ascending = False).head(self.n_best - stay.shape[0])

        df = pd.concat([stay[['secid' , 'indus']] , entry[['secid' , 'indus']]]).\
            assign(weight = 1).filter(items=['secid' , 'weight']).drop_duplicates(subset=['secid'])
        return df

class BasicPortfolioCreator(PortCreator):
    @abstractmethod
    def setup(self , indent : int = 1 , vb_level : Any = 3 , **kwargs):
        self.conf = BasicCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self
    
    def parse_input(self):
        return self

    def solve(self):
        df = self.conf.generate_top_stock_port(
            model_date = self.model_date , 
            alpha_model = self.alpha_model , 
            init_port = self.init_port , 
            bench_port = self.bench_port
        )
        port = Port(df , date = self.model_date , name = self.name , value = self.value).rescale()
        self.create_result = PortCreateResult(port , len(df) == self.conf.n_best)

        return self

    def output(self):
        if self.detail_infos: 
            self.create_result.analyze(self.bench_port , self.init_port)
        return self