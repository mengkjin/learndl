import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from pathlib import Path
from typing import Literal , Type

from src.proj import PATH , Proj , Logger , CALENDAR , DB
from src.proj.func import dfs_to_excel , figs_to_pdf
from src.res.factor.util import StockFactor , Benchmark , Portfolio , AlphaModel , Amodel , Universe
from src.res.factor.fmp import PortfolioBuilder
from src.res.factor.analytic.fmp_top import FrontFace , Perf_Curve , Perf_Excess , Drawdown , Perf_Year , TopCalc
from src.res.factor.calculator import StockFactorHierarchy
from src.res.model.util import PredictionModel

TASK_LIST : list[Type[TopCalc]] = [
    FrontFace , 
    Perf_Curve ,
    Perf_Excess ,
    Drawdown , 
    Perf_Year ,
]

@dataclass
class TradingPort:
    name        : str 
    alpha       : str
    universe    : str
    category    : Literal['top' , 'screen'] = 'top'
    components  : list[str] = field(default_factory=list)
    weights     : list[float] = field(default_factory=list)
    top_num     : int = 50
    freq        : int | Literal['d' , 'w' , 'm'] = 5
    init_value  : float = 1e6
    backtest    : bool = False
    test_start  : int = 20190101 # -1 for no backtest
    test_end    : int = -1
    benchmark   : str = 'csi500'
    sorting_alpha : tuple[str , str , str | None] = ('pred' , 'gru_day_V1' , None)
    screen_ratio : float = 0.5
    buffer_zone : float = 0.8
    no_zone     : float = 0.5
    indus_control : float = 0.1

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

        assert not self.backtest or self.test_start > 0 , f'test_start must be positive when backtest is True: {self.test_start}'
        
        assert self.category in ['top' , 'screen'] , f'category must be top or screen: {self.category}'
        if self.category == 'screen':
            assert self.sorting_alpha is not None , 'sorting_alpha must be provided'
    
        self.test_start = max(self.test_start , 20170101) if self.test_start > 0 else -1
        self.test_end = 20991231 if self.test_end < 0 else self.test_end

        self.Alpha = CompositeAlpha(self.alpha , self.components , self.weights)
        self.Universe = Universe(self.universe)

        self.new_ports : dict[int , pd.DataFrame] = {}
        self.last_ports : dict[int , pd.DataFrame] = {}
    
    @classmethod
    def load(cls , name : str) -> 'TradingPort':
        port_dict = Proj.Conf.TradingPort.portfolio_dict
        assert name in port_dict , f'{name} is not in {port_dict}'
        kwargs = {'name' : name , **port_dict[name]}
        return cls(**kwargs)

    @property
    def export_dir(self) -> Path:
        return PATH.trade_port.joinpath(self.name)
        
    def export_path(self , date : int) -> Path:
        return self.export_dir.joinpath(f'{self.name}.{date}.feather')
    
    @property
    def result_dir(self) -> Path:
        return PATH.rslt_trade.joinpath(self.name)

    @property
    def result_path_account(self) -> Path:
        return self.result_dir.joinpath('account.tar')

    @property
    def result_path_data(self) -> Path:
        return self.result_dir.joinpath(f'{self.name}_analytic_data.xlsx')

    @property
    def result_path_plot(self) -> Path:
        return self.result_dir.joinpath(f'{self.name}_analytic_plot.pdf')
    
    def stored_dates(self , start : int | None = None , end : int | None = None) -> np.ndarray:
        dates = DB.dir_dates(self.export_dir)
        if start is not None: 
            dates = dates[dates >= start]
        if end is not None: 
            dates = dates[dates <= end]
        return dates
    
    def is_first_date(self , date : int) -> bool:
        return self.last_date(date) <= 0

    def last_date(self , date : int) -> int:
        dates = self.stored_dates(end = date - 1)
        return dates.max() if len(dates) > 1 else -1
        
    def start_date(self) -> int:
        dates = self.stored_dates()
        return dates.min() if len(dates) > 1 else -1
    
    def end_date(self) -> int:
        dates = self.stored_dates()
        return dates.max() if len(dates) > 1 else -1
    
    def updatable(self , date : int , force = False) -> bool:
        if force: 
            return True
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
        path = self.export_path(date)
        if path.exists():
            return DB.load_df(path).assign(date = date , name = self.name)
        else:
            return pd.DataFrame()

    def get_last_port(self , date : int , reset_port = False) -> Portfolio:
        if reset_port:
            Logger.alert1(f'Reset port for new build! {self.name}')
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
        assert port.is_empty or max(port.port_date) < date , f'last port date {max(port.port_date)} should be less than date {date}'
        return port
    
    def build(self , date : int , reset = False , export = True , indent : int = 1 , vb_level : int = 2):
        if self.backtest:
            df = self.build_backward(date , reset_port = reset , export = export , indent = indent , vb_level = vb_level)
        else:
            df = self.build_portfolio(date , reset_port = reset , export = export , alpha_details = True , indent = indent , vb_level = vb_level)
        self.new_ports[date] = df
        return self
    
    def build_portfolio(self , date : int , reset_port = False , export = True , last_port = None ,
                        alpha_details = False , indent : int = 1 , vb_level : int = 2) -> pd.DataFrame:
        alpha = self.Alpha.get(date)
        universe = self.Universe.get(date)
        if last_port is None:
            last_port = self.get_last_port(date , reset_port)

        Logger.stdout(f'Perform portfolio building for TradingPort {self.name} on {date}' , indent = indent , vb_level = vb_level)

        builder = PortfolioBuilder(self.category , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , no_zone = self.no_zone , 
                                   indus_control = self.indus_control , sorting_alpha = self.sorting_alpha ,
                                   screen_ratio = self.screen_ratio , indent = indent + 1 , vb_level = vb_level + 1).setup()

        pf = builder.build(date).port.to_dataframe()
        if pf.empty: 
            return pf

        if self.is_first_date(date):
            pf['value'] = self.init_value

        if export:
            self.save_port(pf , date , indent = indent + 1 , vb_level = vb_level + 2)

        # add columns to include alpha and universe
        if alpha_details:
            alpha_model = alpha.item()
            pf['alpha'] = alpha_model.alpha_of(pf['secid'])
            pf['alpha_rank'] = alpha_model.alpha_of(pf['secid'] , rank = True)
        return pf.assign(name = self.name , date = date)
    
    def build_backward(self , date : int , reset_port = False , export = True , indent : int = 1 , vb_level : int = 2) -> pd.DataFrame:
        assert self.backtest , 'backtest must be True'
        assert self.test_start > 0 , 'test_start must be positive'
        test_end = min(date , self.test_end)
        
        if test_end < self.test_start: 
            return pd.DataFrame()

        end_date = self.end_date()
        date_list = CALENDAR.td_within(self.test_start , test_end , self.step)
        if not reset_port:
            date_list = date_list[date_list > end_date]
        if len(date_list) == 0: 
            return pd.DataFrame()
        
        Logger.stdout(f'Perform backtest for TradingPort {self.name} , {len(date_list)} days' , indent = indent , vb_level = vb_level)
        pf = None
        for d in date_list:
            pf = Portfolio.from_dataframe(self.build_portfolio(d , export = export , last_port = pf , indent = indent + 1 , vb_level = vb_level + 1))
        return pd.DataFrame()

    def save_port(self , pf : pd.DataFrame , date : int , indent : int = 1 , vb_level : int = 2):
        path = self.export_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        DB.save_df(pf.loc[:,['secid' , 'weight' , 'value']] , path , prefix = f'Portfolio' , indent = indent , vb_level = vb_level)
    
    def load_portfolio(self , start : int | None = None , end : int | None = None):
        dates = self.stored_dates(start , end)
        #paths = [self.port_path(date) for date in dates]
        #dfs = [DB.load_df(path).assign(date = date) for date , path in zip(dates , paths)]
        #df = pd.concat(dfs).assign(name = self.name)
        df = DB.load_dfs({date:self.export_path(date) for date in dates}).assign(name = self.name)
        self.portfolio = Portfolio.from_dataframe(df , name = self.name)
    
    def portfolio_account(self , start : int = -1 , end : int = 99991231 ,
                          analytic = False , attribution = False , 
                          trade_engine : Literal['default' , 'harvest' , 'yale'] = 'yale' ,
                          indent : int = 1 , vb_level : int = 2):
        self.load_portfolio(start , end)
        benchmark = Benchmark(self.benchmark)
        index = {
            'factor_name' : self.portfolio.name ,
            'benchmark'   : benchmark.name ,
            'strategy'    : f'top{self.top_num}' ,
        }
        self.portfolio.accounting(benchmark , start , end , analytic , attribution , trade_engine = trade_engine ,
                                  resume_path = self.result_path_account , resume_drop_last = True , indent = 1 , vb_level = 2)
        return self.portfolio.account.with_index(index)

    def analyze(self , start : int | None = None , end : int | None = None , 
                indent : int = 0 , vb_level : int = 1 , write_down = False , display_all = False , key_fig = 'top@drawdown', 
                trade_engine : Literal['default' , 'harvest' , 'yale'] = 'yale' , **kwargs):
        if not write_down and not display_all:
            Logger.error(f'write_down and display_all cannot be both False')
            return self
        Logger.stdout(f'Analyze trading portfolio [{self.name}] start ...' , indent = indent , vb_level = vb_level)
        
        start = start if start is not None else -1
        end = end if end is not None else 99991231
        account_df = self.portfolio_account(start = start , end = end , trade_engine=trade_engine , indent = indent + 1 , vb_level = vb_level + 1).df
        candidates = {task.task_name():task for task in TASK_LIST}
        self.tasks = {k:v(**kwargs) for k,v in candidates.items()}
        for task in self.tasks.values():
            task.calc(account_df , indent = indent + 1 , vb_level = vb_level + 2) 
            task.plot(show = False , indent = indent + 1 , vb_level = vb_level + 2)  

        rslts = {k:v.calc_rslt for k,v in self.tasks.items()}
        figs  = {f'{k}@{fig_name}':fig for k,v in self.tasks.items() for fig_name , fig in v.figs.items()}

        if write_down:
            dfs_to_excel(rslts , self.result_path_data , print_prefix=f'Analytic Test of TradingPort {self.name} datas' , indent = indent + 1 , vb_level = vb_level + 2)
            figs_to_pdf(figs   , self.result_path_plot , print_prefix=f'Analytic Test of TradingPort {self.name} plots' , indent = indent + 1 , vb_level = vb_level + 2)

        for name , fig in figs.items():
            if (key_fig and key_fig.lower() in name.lower()) or display_all:
                Logger.display(fig , caption = f'Figure: {name.title()}:')

        self.analyze_results = rslts
        self.analyze_figs = figs
        Logger.success(f'Analyze trading portfolio [{self.name}]!' , indent = indent , vb_level = vb_level)
        return self

@dataclass
class CompositeAlpha:
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
        if alpha_name in PredictionModel.MODEL_DICT:
            reg_model = PredictionModel.SelectModels(alpha_name)[0]
            dates = reg_model.pred_dates
            df = reg_model.load_pred(dates[dates <= date].max()).assign(date = date)
        elif '@' in alpha_name:
            exprs = alpha_name.split('@')
            alpha_type = exprs[0]
            alpha_name = exprs[1]
            alpha_column = exprs[2] if len(exprs) > 2 else alpha_name
            if alpha_type == 'sellside':
                df = DB.load(alpha_type , alpha_name , date , closest = True , vb_level = 99)
            elif alpha_type == 'factor':
                df = StockFactorHierarchy.get_factor(alpha_name).Load(date , closest = True)
            else:
                raise Exception(f'{alpha_type} is not a valid alpha type')
            df = pd.DataFrame(columns=['secid' , 'date' , alpha_column]) if df.empty else df.assign(date = date).loc[:,['secid' , 'date' , alpha_column]]
        else:
            raise Exception(f'{alpha_name} is not a valid alpha')

        factor = StockFactor(df)
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor.normalize().alpha_model()