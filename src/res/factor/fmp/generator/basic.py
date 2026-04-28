import pandas as pd
import numpy as np

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
    def alpha_sorter(self) -> AlphaComposite:
        if not hasattr(self , '_sorter_alpha'):
            self._sorter_alpha = AlphaComposite([alpha for alpha in self.sorter if alpha != 'self'])
        return self._sorter_alpha   
    @property
    def alpha_screener(self) -> AlphaScreener:
        if not hasattr(self , '_screener_alpha'):
            self._screener_alpha = AlphaScreener([alpha for alpha in self.screener if alpha != 'self'] , ratio = self.screen_ratio)
        return self._screener_alpha

    def get_sorting_alpha(self , model_date : int , alpha_model : AlphaModel | Amodel) -> Amodel:
        alpha_composite = self.alpha_sorter.get(model_date , alpha_model if 'self' in self.sorter else None)
        sorting_alpha = alpha_composite.get_model(model_date)
        if not sorting_alpha:
            sorting_alpha = alpha_model.get_model(model_date)
        return sorting_alpha

    def get_candidate_pool(
        self ,
        model_date : int , 
        alpha_model : AlphaModel | Amodel ,
        bench_port : Port | None = None
    ) -> np.ndarray:
        alpha_model = alpha_model if isinstance(alpha_model , AlphaModel) else alpha_model.to_alpha_model()
        candidates = alpha_model.get(model_date).secid
        universe = bench_port.secid if (bench_port and not bench_port.emtpy) else None
        screened = self.alpha_screener.screened_pool(model_date , universe , other_models = alpha_model if 'self' in self.screener else None)
        secid = np.setdiff1d(candidates , screened)
        return secid

    @property
    def stay_num(self) -> float:
        return (1 - self.turn_control) * self.n_best
    
    @property
    def indus_slots(self) -> float:
        return self.n_best * self.indus_control

    def generate_top_stock_port(
        self ,
        model_date : int , 
        alpha_model : AlphaModel | Amodel ,
        init_port : Port | None = None, 
        bench_port : Port | None = None, 
    ) -> pd.DataFrame:
        candidates = self.get_candidate_pool(model_date , alpha_model , bench_port)
        sorting_alpha = self.get_sorting_alpha(model_date , alpha_model).align(candidates)
        pool = sorting_alpha.to_dataframe(indus=True , na_indus_as = 'unknown')
        
        if init_port is None:
            init_port = Port.none_port(model_date)
        
        pool.loc[:, 'ind_rank']  = pool.groupby('indus')['alpha'].rank(method = 'first' , ascending = False)
        pool.loc[:, 'rankpct']   = pool['alpha'].rank(pct = True , method = 'first' , ascending = True)
        
        pool = pool.merge(init_port.port , on = 'secid' , how = 'left').sort_values('alpha' , ascending = False)
        pool.loc[:, 'selected'] = pool['weight'].astype(float).fillna(0) > 0
        pool.loc[:, 'buffered'] = (
            (pool['rankpct'] >= self.buffer_zone) + (pool['selected'].cumsum() <= self.stay_num) * (pool['rankpct'] >= self.no_zone))

        stay = pool.query('selected & buffered')

        remain = pool.query('secid not in @stay.secid')
        remain = remain.merge(stay.groupby('indus')['secid'].count().rename('ind_count') , on = 'indus' , how = 'left')
        remain['max_ind_rank'] = self.indus_slots - remain['ind_count'].fillna(0)

        entry = remain.query('ind_rank < max_ind_rank').sort_values('alpha' , ascending = False).head(self.n_best - stay.shape[0])

        df = pd.DataFrame({'secid' : np.union1d(stay['secid'] , entry['secid']) , 'weight' : 1})
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