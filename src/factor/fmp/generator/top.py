import pandas as pd

from dataclasses import dataclass

from src.factor.util import Port , PortCreateResult , PortCreator

DEFAULT_N_BEST = 50

@dataclass(slots = True)
class PortfolioGeneratorConfig:
    '''
    Config for PortfolioGenerator (Top Portfolio Generator)
    kwargs:
        n_best          : int = DEFAULT_N_BEST , number of best stocks to select
        turn_control    : float = 0.2 , control the number of stocks to be substituted
        buffer_zone     : float = 0.8 , above this rankpct, the stock will remain in the portfolio
        no_zone         : float = 0.5 , below this rankpct, the stock will not be substituted
        indus_control   : float = 0.1 , control the ratio of stocks to select in one industry
    '''
    n_best          : int = DEFAULT_N_BEST
    turn_control    : float = 0.2
    buffer_zone     : float = 0.8
    no_zone         : float = 0.5
    indus_control   : float = 0.1

    @classmethod
    def init_from(cls , print_info : bool = False , **kwargs):
        use_kwargs = {k: v for k, v in kwargs.items() if k in cls.__slots__ and v is not None}
        drop_kwargs = {k: v for k, v in kwargs.items() if k not in cls.__slots__}
        if print_info:
            if use_kwargs : print(f'In initializing {cls.__name__}, used kwargs: {use_kwargs}')
            if drop_kwargs: print(f'In initializing {cls.__name__}, dropped kwargs: {drop_kwargs}')
        return cls(**use_kwargs)
    
    @property
    def stay_num(self):
        return (1 - self.turn_control) * self.n_best
    
    @property
    def indus_slots(self):
        return self.n_best * self.indus_control
    
class PortfolioGenerator(PortCreator):
    DEFAULT_N_BEST = DEFAULT_N_BEST
    def __init__(self , name : str):
        super().__init__(name)

    def setup(self , print_info : bool = False , **kwargs):
        self.conf = PortfolioGeneratorConfig.init_from(print_info = print_info , **kwargs)
        return self
    
    def parse_input(self):
        return self

    def solve(self):
        pool = self.alpha_model.get_model(self.model_date).to_dataframe(indus=True , na_indus_as = 'unknown')
        if not self.bench_port.is_emtpy(): pool = pool.loc[pool['secid'].isin(self.bench_port.secid)]
        pool['ind_rank']  = pool.groupby('indus')['alpha'].rank(method = 'first' , ascending = False)
        pool['rankpct']   = pool['alpha'].rank(pct = True , method = 'first' , ascending = True)
        
        pool = pool.merge(self.init_port.port , on = 'secid' , how = 'left').sort_values('alpha' , ascending = False)
        pool['selected'] = pool['weight'].fillna(0) > 0
        pool['buffered'] = ((pool['rankpct'] >= self.conf.buffer_zone) + (pool['selected'].cumsum() <= self.conf.stay_num) *
                            (pool['rankpct'] >= self.conf.no_zone))

        stay = pool[pool['selected'] & pool['buffered']]
        stay_ind_count = stay.groupby('indus')['secid'].count().rename('count')

        new_pool = pool[~pool['secid'].isin(stay['secid'])].merge(stay_ind_count , on = 'indus' , how = 'left')
        selectable = new_pool['ind_rank'] < (self.conf.indus_slots - new_pool['count'].fillna(0))
        entry = new_pool[selectable].sort_values('alpha' , ascending = False).head(self.conf.n_best - stay.shape[0])

        df = pd.concat([stay[['secid' , 'indus']] , entry[['secid' , 'indus']]])
        df = df.drop_duplicates(subset=['secid']).assign(weight = 1)[['secid' , 'weight']]
        port = Port(df , date = self.model_date , name = self.name , value = self.value).rescale()
        self.create_result = PortCreateResult(port , len(df) == self.conf.n_best)

        return self

    def output(self):
        if self.detail_infos: self.create_result.analyze(self.bench_port , self.init_port)
        return self