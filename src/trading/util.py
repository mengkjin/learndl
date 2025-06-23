import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from pathlib import Path
from typing import Literal , Type , Any

from src.basic import CALENDAR , RegisteredModel , PATH , CONF
from src.data import DATAVENDOR
from src.func import display as disp
from src.factor.util import StockFactor , Benchmark , Portfolio , AlphaModel , Amodel , Port
from src.factor.fmp import PortfolioBuilder
from src.factor.analytic.fmp_top.api import Calc
from src.func import dfs_to_excel , figs_to_pdf

TASK_LIST : list[Type[Calc.BaseTopPortCalc]] = [
    Calc.Top_FrontFace , 
    Calc.Top_Perf_Curve ,
    Calc.Top_Perf_Excess ,
    Calc.Top_Perf_Drawdown , 
    Calc.Top_Perf_Year ,
]

def all_path_convert():
    file_list = list(PATH.trade_port.glob("**/*.csv"))
    if not file_list:
        return
    print(f'Converting {len(file_list)} csv files to feather files')
    for path in file_list:
        new_path = path.with_suffix('.feather')
        df = pd.read_csv(path)
        df.to_feather(new_path)
        path.unlink()

all_path_convert()

@dataclass
class TradingPort:
    name        : str 
    alpha       : str
    universe    : str
    components  : list[str] = field(default_factory=list)
    weights     : list[float] = field(default_factory=list)
    top_num     : int = 50
    freq        : int | Literal['d' , 'w' , 'm'] = 5
    init_value  : float = 1e6
    backtest    : bool = False
    test_start  : int = 20190101 # -1 for no backtest
    test_end    : int = -1
    benchmark   : str = 'csi500'

    def __post_init__(self):
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
        self.no_zone = 0.5
        self.indus_control = 0.1

        assert not self.backtest or self.test_start > 0 , f'test_start must be positive when backtest is True: {self.test_start}'
        self.test_start = max(self.test_start , 20170101) if self.test_start > 0 else -1
        self.test_end = 20991231 if self.test_end < 0 else self.test_end

        self.Alpha = Alpha(self.alpha , self.components , self.weights)
        self.Universe = Universe(self.universe)

        self.new_ports : dict[int , pd.DataFrame] = {}
        self.last_ports : dict[int , pd.DataFrame] = {}

    @classmethod
    def portfolio_dict(cls) -> dict[str , dict]:
        return CONF.trade('portfolio_dict')
    
    @classmethod
    def load(cls , name : str) -> 'TradingPort':
        port_dict = cls.portfolio_dict()
        assert name in port_dict , f'{name} is not in portfolio_dict'
        return cls(name , **port_dict[name])

    def port_dir(self) -> Path:
        return PATH.trade_port.joinpath(self.name)
        
    def port_path(self , date : int) -> Path:
        return self.port_dir().joinpath(f'{self.name}.{date}.feather')
    
    def result_dir(self) -> Path:
        return PATH.rslt_trade.joinpath(self.name)
    
    def existing_dates(self , min_date : int | None = None , max_date : int | None = None) -> np.ndarray:
        dates = PATH.dir_dates(self.port_dir())
        if min_date is not None: dates = dates[dates >= min_date]
        if max_date is not None: dates = dates[dates <= max_date]
        return dates
    
    def is_first_date(self , date : int) -> bool:
        return self.last_date(date) <= 0

    def last_date(self , date : int) -> int:
        dates = self.existing_dates(max_date = date - 1)
        return dates.max() if len(dates) > 1 else -1
        
    def start_date(self) -> int:
        dates = self.existing_dates()
        return dates.min() if len(dates) > 1 else -1
    
    def end_date(self) -> int:
        dates = self.existing_dates()
        return dates.max() if len(dates) > 1 else -1
    
    def updatable(self , date : int , force = False) -> bool:
        if force: return True
        if self.backtest:
            if self.test_end > 0 and date > self.test_end: 
                return False
            if self.test_start > 0 and date < self.test_start:
                return False
        last_date = self.last_date(date)
        if last_date < 0: 
            return True
        else:
            return CALENDAR.td(last_date , self.step) <= date
    
    def load_port(self , date : int) -> pd.DataFrame:
        path = self.port_path(date)
        if path.exists():
            return pd.read_feather(path).assign(date = date , name = self.name)
        else:
            return pd.DataFrame()

    def get_last_port(self , date : int , reset_port = False) -> Portfolio:
        if reset_port:
            print(f'Beware: reset port for new build! {self.name}')
            port = Portfolio(self.name)
        else:
            if date in self.last_ports:
                port = Portfolio.from_dataframe(self.last_ports[date])
            else:
                if (last_date := self.last_date(date)) > 0:
                    df = self.load_port(last_date)
                    self.last_ports[last_date] = df
                    port = Portfolio.from_dataframe(df)
                else:
                    port = Portfolio(self.name)
        return port
    
    def build(self , date : int , reset = False , export = True):
        if self.backtest:
            df = self.build_backward(date , reset_port = reset , export = export)
        else:
            df = self.build_portfolio(date , reset_port = reset , export = export)
        self.new_ports[date] = df
        return self
    
    def build_portfolio(self , date : int , reset_port = False , export = True , last_port = None) -> pd.DataFrame:
        alpha = self.Alpha.get(date)
        universe = self.Universe.get(date)
        if last_port is None:
            last_port = self.get_last_port(date , reset_port)

        builder = PortfolioBuilder('top' , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , no_zone = self.no_zone , 
                                   indus_control = self.indus_control).setup(0)

        pf = builder.build(date).port.to_dataframe()
        if pf.empty: return pf

        if self.is_first_date(date):
            pf['value'] = self.init_value

        if export:
            path = self.port_path(date)
            path.parent.mkdir(parents=True, exist_ok=True)
            pf.loc[:,['secid' , 'weight' , 'value']].to_feather(path)

        return pf
    
    def build_backward(self , date : int , reset_port = False , export = True) -> pd.DataFrame:
        assert self.backtest , 'backtest must be True'
        assert self.test_start > 0 , 'test_start must be positive'
        test_end = min(date , self.test_end)
        
        if test_end < self.test_start: return pd.DataFrame()

        end_date = self.end_date()
        date_list = CALENDAR.td_within(self.test_start , test_end , self.step)
        if not reset_port:
            date_list = date_list[date_list > end_date]
        if len(date_list) == 0: return pd.DataFrame()
        
        print(f'Perform backtest for TradingPort {self.name} , {len(date_list)} days')
        pf = None
        for d in date_list:
            df = self.build_portfolio(d , reset_port = False , export = export , last_port = pf)
            pf = Portfolio.from_dataframe(df.assign(date = d , name = self.name))

        return pd.DataFrame()
    
    def load_portfolio(self , start : int | None = None , end : int | None = None) -> Portfolio:
        dates = self.existing_dates(start , end)
        paths = [self.port_path(date) for date in dates]
        dfs = [pd.read_feather(path).assign(date = date) for date , path in zip(dates , paths)]
        df = pd.concat(dfs).assign(name = self.name)
        return Portfolio.from_dataframe(df , name = self.name)
    
    def portfolio_account(self , start : int = -1 , end : int = 99991231 ,
                          analytic = False , attribution = False , 
                          trade_engine : Literal['default' , 'harvest' , 'yale'] = 'yale') -> pd.DataFrame:
        portfolio = self.load_portfolio(start , end)
        benchmark = Benchmark(self.benchmark)
        default_index = {
            'factor_name' : portfolio.name ,
            'benchmark'   : benchmark.name ,
            'strategy'    : f'top{self.top_num}' ,
        }
        portfolio.accounting(benchmark , start , end , analytic , attribution , trade_engine)
        self.portfolio = portfolio
        return portfolio.account_with_index(default_index)

    def analyze(self , start : int = -1 , end : int = 99991231 , 
                verbosity = 1 , write_down = False , display = True , 
                trade_engine : Literal['default' , 'harvest' , 'yale'] = 'yale' , **kwargs):
        if not write_down and not display:
            print('write_down and display cannot be both False')
            return self

        account = self.portfolio_account(start = start , end = end , trade_engine=trade_engine)
        candidates = {task.task_name():task for task in TASK_LIST}
        self.tasks = {k:v(**kwargs) for k,v in candidates.items()}
        for task in self.tasks.values():
            task.calc(account , verbosity = verbosity - 1) 
            task.plot(show = False)  

        rslts = {k:v.calc_rslt for k,v in self.tasks.items()}
        figs  = {f'{k}@{fig_name}':fig for k,v in self.tasks.items() for fig_name , fig in v.figs.items()}

        if write_down:
            dfs_to_excel(rslts , self.result_dir().joinpath('data.xlsx') , print_prefix=f'Analytic Test of TradingPort {self.name} datas')
            figs_to_pdf(figs   , self.result_dir().joinpath('plot.pdf')  , print_prefix=f'Analytic Test of TradingPort {self.name} plots')

        if display:
            [disp.plot(fig) for fig in figs.values()]

        self.analyze_results = rslts
        self.analyze_figs = figs
        if verbosity > 0: print(f'{self.name} analyze Finished!')
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