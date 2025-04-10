import pandas as pd

from dataclasses import dataclass , field

from src.basic import CALENDAR , RegisteredModel , PATH 
from src.data import DATAVENDOR
from src.factor.util import StockFactor , Benchmark , Portfolio , AlphaModel , Amodel , Port
from src.factor.fmp import PortfolioBuilder

@dataclass
class TradingPort:
    name        : str 
    alpha       : str = ''
    universe    : str = 'all'
    components  : list[str] = field(default_factory=list)
    weights     : list[float] = field(default_factory=list)
    top_num     : int = 50
    step        : int = 5
    init_value  : float = 1e6

    @classmethod
    def load(cls , port_name : str , **kwargs):
        return cls(port_name , **kwargs)

    def __post_init__(self):
        self.get_params()

    def get_params(self):
        assert self.step > 0 , 'step must be positive'
        if self.step == 1:
            self.turn_control = 0.1
        elif self.step <= 4:
            self.turn_control = 0.1
        elif self.step <= 10:
            self.turn_control = 0.2
        elif self.step <= 30:
            self.turn_control = 0.5
        else:
            self.turn_control = 1.0
        self.buffer_zone = 0.8
        self.indus_control = 0.1

    def first_date(self , date : int) -> int:
        return self.last_date(date) <= 0
    
    def last_date(self , date : int) -> int:
        dates = PATH.dir_dates(PATH.trade_port.joinpath(self.name))
        dates = dates[dates < date]
        if len(dates) > 0:
            return dates.max()
        else:
            return -1
        
    def port_path(self , date : int):
        return PATH.trade_port.joinpath(self.name , f'{self.name}.{date}.csv')
        
    def updatable(self , date : int , force = False) -> bool:
        if force: return True
        if (last_date := self.last_date(date)) < 0: return True
        return CALENDAR.td(last_date , self.step) <= date

    def get_last_port(self , date : int) -> Portfolio:
        if (last_date := self.last_date(date)) > 0:
            df = pd.read_csv(self.port_path(last_date)).assign(date = last_date , name = self.name)
            port = Portfolio.from_dataframe(df)
        else:   
            port = Portfolio(self.name)
        return port
    
    def get_alpha(self , date : int) -> AlphaModel:
        return Alpha(self.alpha , self.components , self.weights).get(date)
    
    def get_universe(self , date : int) -> Portfolio:
        return Universe(self.universe).get(date)
    
    def get_builder(self , date : int , reset_port = False) -> PortfolioBuilder:
        alpha = self.get_alpha(date)
        universe = self.get_universe(date)
        if reset_port:
            print(f'Beware: reset port for new build! {self.name}')
            last_port = Portfolio(self.name)
        else:
            last_port = self.get_last_port(date)
        builder = PortfolioBuilder('top' , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , indus_control = self.indus_control).setup(0)
        return builder

@dataclass
class Alpha:
    name        : str
    components  : list[str] = field(default_factory=list)
    weights     : list[float] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.name) > 0 or len(self.components) > 0 , 'name or components must be non-empty'
        assert len(self.components) == len(self.weights) or len(self.weights) == 0 , 'components and weights must have the same length'
    
    def get(self , date : int) -> AlphaModel:
        if len(self.components) == 0:
            return self.get_single_alpha(self.name , date)
        else:
            alphas = [self.get_single_alpha(alpha , date).item() for alpha in self.components]
            amodel = Amodel.combine(alphas , self.weights)
            return AlphaModel(self.name , amodel)
    
    @staticmethod
    def get_single_alpha(alpha_name : str , date : int) -> AlphaModel:
        if alpha_name in RegisteredModel.MODEL_DICT:
            reg_model = RegisteredModel.SelectModels(alpha_name)[0]
            dates = reg_model.pred_dates
            df = reg_model.load_pred(dates[dates <= date].max())
        elif '@' in alpha_name:
            exprs = alpha_name.split('@')
            if exprs[0] == 'sellside':
                dates = PATH.db_dates('sellside' , exprs[1])
                df = PATH.db_load('sellside' , exprs[1] , dates[dates <= date].max() , verbose = False).loc[:,['secid' , exprs[2]]]
            else:
                raise Exception(f'{alpha_name} is not a valid alpha')
        else:
            raise Exception(f'{alpha_name} is not a valid alpha')
        factor = StockFactor(df.assign(date = date))
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor.normalize().alpha_model()
    
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
            val = val[val['secid'].isin(candidates)].iloc[:top_num].loc[:,['secid']].\
                reset_index().assign(date = date , name = self.name)
            val['weight'] = 1 / len(val)
            pf = Portfolio.from_dataframe(val , name = self.name)
        elif self.name in Benchmark.AVAILABLES:
            pf = Benchmark(self.name)
        elif '+' in self.name:
            univs = [Benchmark(univ).get(date) for univ in self.name.split('+')]
            pf = Portfolio.from_ports(Port.sum(univs) , name = self.name)
        else:
            raise Exception(f'{self.universe} is not a valid benchmark')
        
        assert isinstance(pf , Portfolio) , f'expect Portfolio , got {type(pf)}'

        if safety:
            st_list = DATAVENDOR.INFO.get_st(date)['secid'].to_numpy()
            small_cp = DATAVENDOR.TRADE.get_val(date).query('close < 2.0')['secid'].to_numpy()

            pf = pf.exclude(st_list , True).exclude(small_cp , True)

        return pf