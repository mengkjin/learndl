import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from pathlib import Path
from typing import Literal , Type , Any

from src.basic import CALENDAR , RegisteredModel , PATH , CONF
from src.data import DATAVENDOR
from src.factor.util import StockFactor , Benchmark , Portfolio , AlphaModel , Amodel , Port
from src.factor.fmp import PortfolioBuilder
from src.factor.analytic.fmp_top.api import Calc
from src.factor.fmp.accountant import portfolio_account

TASK_LIST : list[Type[Calc.BaseTopPortCalc]] = [
    Calc.Top_FrontFace , 
    Calc.Top_Perf_Curve ,
    Calc.Top_Perf_Excess ,
    Calc.Top_Perf_Drawdown , 
    Calc.Top_Perf_Year ,
]

@dataclass
class TradingPort:
    name        : str 
    alpha       : str = ''
    universe    : str = 'all'
    components  : list[str] = field(default_factory=list)
    weights     : list[float] = field(default_factory=list)
    top_num     : int = 50
    freq        : int | Literal['d' , 'w' , 'm'] = 5
    init_value  : float = 1e6
    backtest    : bool = False
    test_start  : int = 20190101 # -1 for no backtest
    benchmark   : str = 'CSI500'

    def __post_init__(self):
        self.init_params()

        self.Alpha = Alpha(self.alpha , self.components , self.weights)
        self.Universe = Universe(self.universe)

    def init_params(self):
        assert isinstance(self.freq , int) or self.freq in ['d' , 'w' , 'm'] , f'str freq must be d, w, or m : {self.freq}'
        self.step = self.freq if isinstance(self.freq , int) else {'d' : 1 , 'w' : 5 , 'm' : 20}[self.freq]

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

        self.test_start = max(self.test_start , 20170101) if self.test_start > 0 else -1
    
    @classmethod
    def portfolio_dict(cls) -> dict[str , dict]:
        return CONF.trade('portfolio_dict')
    
    @classmethod
    def load(cls , name : str) -> 'TradingPort':
        port_dict = cls.portfolio_dict()
        assert name in port_dict , f'{name} is not in portfolio_dict'
        return cls(**port_dict[name])

    @property
    def backtest_name(self) -> str:
        return f'{self.name}.backtest'
    
    @property
    def is_backtested(self) -> bool:
        return self.port_dir().exists()
    
    def port_name(self , backtest = False) -> str:
        return self.backtest_name if backtest else self.name

    def is_first_date(self , date : int , backtest = False) -> bool:
        return self.last_date(date , backtest) <= 0
    
    def existing_dates(self , min_date : int | None = None , max_date : int | None = None , backtest = False) -> np.ndarray:
        dates = PATH.dir_dates(self.port_dir(backtest))
        if min_date is not None: dates = dates[dates >= min_date]
        if max_date is not None: dates = dates[dates <= max_date]
        return dates
    
    def last_date(self , date : int , backtest = False) -> int:
        dates = self.existing_dates(max_date = date - 1 , backtest = backtest)
        return dates.max() if len(dates) > 1 else -1
        
    def start_date(self , backtest = False) -> int:
        dates = self.existing_dates(backtest = backtest)
        return dates.min() if len(dates) > 1 else -1
    
    def end_date(self , backtest = False) -> int:
        dates = self.existing_dates(backtest = backtest)
        return dates.max() if len(dates) > 1 else -1
    
    def port_dir(self , backtest = False) -> Path:
        return PATH.trade_port.joinpath(self.port_name(backtest))
        
    def port_path(self , date : int , backtest = False) -> Path:
        return self.port_dir(backtest).joinpath(f'{self.port_name(backtest)}.{date}.csv')
        
    def updatable(self , date : int , force = False) -> bool:
        if force: return True
        if (last_date := self.last_date(date)) < 0: return True
        return CALENDAR.td(last_date , self.step) <= date
    
    def load_port(self , date : int , backtest = False) -> pd.DataFrame:
        path = self.port_path(date , backtest)
        if path.exists():
            return pd.read_csv(path).assign(date = date , name = self.port_name(backtest))
        else:
            return pd.DataFrame()

    def get_last_port(self , date : int , reset_port = False , backtest = False) -> Portfolio:
        if not reset_port and (last_date := self.last_date(date , backtest)) > 0:
            df = self.load_port(last_date , backtest)
            port = Portfolio.from_dataframe(df)
        else:
            if reset_port:
                print(f'Beware: reset port for new build! {self.port_name(backtest)}')
            port = Portfolio(self.port_name(backtest))
        return port
    
    def go_backtest(self , test_end : int | Any = None) -> 'TradingPort':
        if not self.backtest or self.test_start <= 0:
            return self
        if test_end is None:
            if (start_date := self.start_date()) > 0:
                test_end = CALENDAR.td(start_date , -1)
            else:
                test_end = CALENDAR.updated()
        if test_end < self.test_start:
            return self

        date_list = CALENDAR.td_within(self.test_start , test_end , self.step)
        existing_dates = self.existing_dates(backtest = True)
        date_list = np.setdiff1d(date_list , existing_dates)
        if len(date_list):
            print(f'Perform backtest for TradingPort {self.name} , {len(date_list)} days')
        for date in date_list:
            self.build_portfolio(date , reset_port = False , export = True , backtest = True)

        return self
    
    def build_portfolio(self , date : int , reset_port = False , export = True , backtest = False) -> pd.DataFrame:
        alpha = self.Alpha.get(date)
        universe = self.Universe.get(date)
        last_port = self.get_last_port(date , reset_port , backtest)
        builder = PortfolioBuilder('top' , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , indus_control = self.indus_control).setup(0)

        pf = builder.build(date).port.to_dataframe()
        if pf.empty: return pf

        if self.is_first_date(date):
            pf['value'] = self.init_value

        if export:
            path = self.port_path(date , backtest)
            path.parent.mkdir(parents=True, exist_ok=True)
            pf.loc[:,['secid' , 'weight' , 'value']].to_csv(path)

        return pf

    
    def load_portfolio(self , start : int | None = None , end : int | None = None , backtest = False) -> Portfolio:
        if start is None:
            start = self.start_date(backtest)
        if end is None:
            end = self.end_date(backtest)
        dates = self.existing_dates(start , end , backtest)
        dfs = [self.load_port(date , backtest) for date in dates]
        return Portfolio.from_dataframe(pd.concat(dfs) , name = self.port_name(backtest))
    
    def portfolio_account(self , start : int = -1 , end : int = 99991231 , backtest = False ,
                          analytic = False , attribution = False) -> pd.DataFrame:
        port = self.load_portfolio(start , end , backtest)
        benchmark = Benchmark(self.benchmark)
        default_index = {
            'factor_name' : port.name ,
            'benchmark'   : benchmark.name ,
            'strategy'    : f'top{self.top_num}' ,
        }
        return portfolio_account(port , benchmark , start , end , 
                                 analytic = analytic , attribution = attribution , index = default_index)

    def analyze(self , start : int = -1 , end : int = 99991231 , backtest = False , verbosity = 1 , **kwargs):
        account = self.portfolio_account(start = start , end = end , backtest = backtest)
        candidates = {task.task_name():task for task in TASK_LIST}
        self.tasks = {k:v(**kwargs) for k,v in candidates.items()}
        for task in self.tasks.values():
            task.calc(account , verbosity = verbosity - 1) 
        if verbosity > 0: print(f'{self.port_name(backtest)} analyze Finished!')
        return self

@dataclass
class Alpha:
    name        : str
    components  : list[str] = field(default_factory=list)
    weights     : list[float] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.name) > 0 or len(self.components) > 0 , 'name or components must be non-empty'
        assert len(self.components) == len(self.weights) or len(self.weights) == 0 , 'components and weights must have the same length'
    
    def get_alphas(self , date : int) -> list[AlphaModel]:
        if len(self.components) == 0:
            return [self.get_single_alpha(self.name , date)]
        else:
            return [self.get_single_alpha(alpha , date) for alpha in self.components]

    def get(self , date : int) -> AlphaModel:
        if len(self.components) == 0:
            return self.get_single_alpha(self.name , date)
        else:
            alphas = self.get_alphas(date)
            amodel = Amodel.combine([alphas.item() for alphas in alphas] , self.weights)
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
            raise Exception(f'{self.name} is not a valid benchmark')
        
        assert isinstance(pf , Portfolio) , f'expect Portfolio , got {type(pf)}'

        if safety:
            st_list = DATAVENDOR.INFO.get_st(date)['secid'].to_numpy()
            small_cp = DATAVENDOR.TRADE.get_val(date).query('close < 2.0')['secid'].to_numpy()

            pf = pf.exclude(st_list , True).exclude(small_cp , True)

        return pf