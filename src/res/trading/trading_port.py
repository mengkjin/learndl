import pandas as pd
import numpy as np
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any , Literal , Type , ClassVar

from src.proj import PATH , Proj , Logger , CALENDAR , DB , Dates
from src.proj.func import dfs_to_excel , figs_to_pdf
from src.res.factor.util import Benchmark , Portfolio , CompositeAlpha , Universe , Port
from src.res.factor.fmp import PortfolioBuilder
from src.res.factor.analytic.fmp_top import FrontFace , Perf_Curve , Perf_Excess , Drawdown , Perf_Year , TopCalc

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
    alpha       : str | list[str]
    universe    : str = 'top-1000'
    category    : Literal['top' , 'screen' , 'reinforce'] = 'top'
    components  : list[str] | None = None
    weights     : list[float] | Literal['equal'] | None = None
    top_num     : int = 50
    freq        : int | Literal['d' , 'w' , 'm'] = 1
    init_value  : float = 1e6
    backtest    : bool = False
    test_start  : int = 20190101 # -1 for no backtest
    test_end    : int = -1
    benchmark   : str = 'csi500'
    exclusion   : str = 'st_bse_lowprice_loser'
    sorting_alpha : tuple[str , str , str | None] = ('pred' , 'gru_day_V1' , None)
    screen_ratio  : float = 0.5
    buffer_zone   : float = 0.8
    no_zone       : float = 0.5
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
        
        assert self.category in ['top' , 'screen' , 'reinforce'] , f'category must be top or screen or reinforce: {self.category}'
        if self.category in ['screen' , 'reinforce']:
            assert self.sorting_alpha is not None , 'sorting_alpha must be provided'
    
        self.test_start = max(self.test_start , 20170101) if self.test_start > 0 else -1
        self.test_end = 20991231 if self.test_end < 0 else self.test_end

        self.Alpha = CompositeAlpha(self.alpha , self.components , self.weights)
        self.Universe = Universe(self.universe)

        self.new_ports : dict[int , pd.DataFrame] = {}
        self.last_ports : dict[int , pd.DataFrame] = {}
    
    @classmethod
    def load(cls , name : str) -> 'TradingPort':
        if name in TrackingPort.candidate_ports and name in BacktestPort.candidate_ports:
            raise ValueError(f'{name} is both tracking and backtest port, please use distinct name for tracking and backtest ports')
        elif name in TrackingPort.candidate_ports:
            return TrackingPort.load(name)
        elif name in BacktestPort.candidate_ports:
            return BacktestPort.load(name)
        else:
            raise ValueError(f'{name} is not in {TrackingPort.candidate_ports.keys()} or {BacktestPort.candidate_ports.keys()}')

    @property
    def result_dir(self) -> Path:
        raise NotImplementedError(f'{self.__class__.__name__} must implement result_dir in subclass')

    @property
    def portfolio_dir(self) -> Path:
        raise NotImplementedError(f'{self.__class__.__name__} must implement portfolio_dir in subclass')

    def build(self , date : int | None = None , reset = False , export = True , indent : int = 1 , vb_level : Any = 2) -> 'TradingPort':
        raise NotImplementedError(f'{self.__class__.__name__} must implement build in subclass')

    def rebuild(self , date : int | None = None , export = True , indent : int = 1 , vb_level : Any = 2) -> 'TradingPort':
        raise NotImplementedError(f'{self.__class__.__name__} must implement rebuild in subclass')
        
    def export_path(self , date : int) -> Path:
        return self.portfolio_dir.joinpath(f'{self.name}.{date}.feather')

    @property
    def result_path_account(self) -> Path:
        return self.result_dir.joinpath('account.tar')

    @property
    def result_path_data(self) -> Path:
        return self.result_dir.joinpath(f'{self.name}_analytic_data.xlsx')

    @property
    def result_path_plot(self) -> Path:
        return self.result_dir.joinpath(f'{self.name}_analytic_plot.pdf')

    @property
    def trading_portfolio_type(self) -> Literal['tracking' , 'backtest']:
        if self.backtest:
            return 'backtest'
        else:
            return 'tracking'
    
    def stored_dates(self , start : int | None = None , end : int | None = None) -> np.ndarray:
        dates = DB.dir_dates(self.portfolio_dir , start_dt = start , end_dt = end)
        return dates
    
    def is_first_date(self , date : int) -> bool:
        return self.last_date(date) <= 0

    def last_date(self , date : int) -> int:
        dates = self.stored_dates(end = date - 1)
        return dates.max() if len(dates) > 0 else -1
        
    def start_date(self) -> int:
        dates = self.stored_dates()
        return dates.min() if len(dates) > 0 else -1
    
    def end_date(self) -> int:
        dates = self.stored_dates()
        return dates.max() if len(dates) > 0 else -1
    
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
            port = self.empty_pre_portfolio(date)
        else:
            if date in self.last_ports:
                port = Portfolio.from_dataframe(self.last_ports[date])
            else:
                if (last_date := self.last_date(date)) > 0:
                    df = self.load_port(last_date)
                    self.last_ports[last_date] = df
                    port = Portfolio.from_dataframe(df)
                else:
                    port = self.empty_pre_portfolio(date)
        assert port.empty or max(port.port_date) < date , f'last port date {max(port.port_date)} should be less than date {date}'
        return port

    def empty_pre_portfolio(self , date : int) -> Portfolio:
        return Portfolio.from_ports(Port.none_port(CALENDAR.td(date , -1).td , self.name , self.init_value))
 
    def save_port(self , pf : pd.DataFrame , date : int , indent : int = 1 , vb_level : Any = 2):
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
                          indent : int = 1 , vb_level : Any = 2):
        self.load_portfolio(start , end)
        benchmark = Benchmark(self.benchmark)
        index = {
            'factor_name' : self.portfolio.name ,
            'benchmark'   : benchmark.name ,
            'strategy'    : f'top{self.top_num}' ,
        }
        self.portfolio.accounting(benchmark , start , end , analytic , attribution , trade_engine = trade_engine ,
                                  with_index = index , resume_path = self.result_path_account , resume_drop_last = True , 
                                  save_after = True , indent = indent , vb_level = vb_level)
        return self.portfolio.account

    def analyze(self , start : int | None = None , end : int | None = None , 
                indent : int = 0 , vb_level : Any = 1 , write_down = True , display_all = False , key_fig = 'perf_curve', 
                trade_engine : Literal['default' , 'harvest' , 'yale'] = 'yale' , **kwargs):
        vb_level = Proj.vb(vb_level)
        start = start if start is not None else -1
        end = end if end is not None else 99991231
        port_dates = self.stored_dates(start , end)
        if len(port_dates) == 0:
            Logger.alert1(f'No portfolio dates found for {self.name} between {start} and {end} , call build(end_date) first!')
            return self

        Logger.stdout(f'Analyze {self.trading_portfolio_type.title()} Portfolio [{self.name}] at {Dates(port_dates)} start ...' , indent = indent , vb_level = vb_level)
        account_df = self.portfolio_account(start = start , end = end , trade_engine=trade_engine , indent = indent + 1 , vb_level = vb_level + 2).df
        if len(account_df) <= 1:
            Logger.stdout(f'{self.trading_portfolio_type.title()} Portfolio [{self.name}] just start accounting and has no record' , indent = indent , vb_level = vb_level + 1)
            return self
        
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
                Logger.display(fig , caption = f'Figure: {name.title()}:' , vb_level = vb_level)

        self.analyze_results = rslts
        self.analyze_figs = figs
        Logger.success(f'Analyze {self.trading_portfolio_type.title()} Portfolio [{self.name}]!' , indent = indent , vb_level = vb_level)
        return self

class TrackingPort(TradingPort):
    candidate_ports : ClassVar[dict[str , dict]] = Proj.Conf.TradingPort.tracking_ports
    @classmethod
    def load(cls , name : str) -> 'TrackingPort':
        if name in cls.candidate_ports:
            kwargs = {'name' : name , **cls.candidate_ports[name]} | {'backtest' : False}
            return cls(**kwargs)
        else:
            raise ValueError(f'{name} is not a valid tracking port , find configs/setting/trading_port.yaml for available tracking ports')

    @property
    def result_dir(self) -> Path:
        return PATH.rslt_trade.joinpath('tracking' , self.name)

    @property
    def portfolio_dir(self) -> Path:
        return PATH.trade_port.joinpath(self.name)
    
    def build(self , date : int | None = None , reset = False , export = True , indent : int = 1 , vb_level : Any = 2):
        date = CALENDAR.updated(date)
        df = self.build_portfolio(date , reset_port = reset , export = export , alpha_details = True , indent = indent , vb_level = vb_level)
        self.new_ports[date] = df
        return self

    def rebuild(self , date : int , export = True , indent : int = 1 , vb_level : Any = 2):
        raise TypeError(f'tracking port cannot rebuild. if you truely want to rebuild a tracking port, manually delete the portfolio folder and run build(end_date) again.')
    
    def build_portfolio(self , date : int , reset_port = False , export = True , last_port = None ,
                        alpha_details = False , indent : int = 1 , vb_level : Any = 2) -> pd.DataFrame:
        vb_level = Proj.vb(vb_level)
        alpha = self.Alpha.get(date)
        universe = self.Universe.get(date , self.exclusion)
        if last_port is None:
            last_port = self.get_last_port(date , reset_port)

        Logger.stdout(f'Perform portfolio building for TradingPort {self.name} at {Dates(date)}' , indent = indent , vb_level = vb_level)
        builder = PortfolioBuilder(self.category , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , no_zone = self.no_zone , 
                                   indus_control = self.indus_control , sorting_alpha = self.sorting_alpha ,
                                   screen_ratio = self.screen_ratio , indent = indent + 1 , vb_level = vb_level + 1).setup()

        pf = builder.build(date).port.to_dataframe()
        if pf.empty: 
            return pf

        if export:
            self.save_port(pf , date , indent = indent + 1 , vb_level = vb_level + 2)

        # add columns to include alpha and universe
        if alpha_details:
            alpha_model = alpha.item()
            pf['alpha'] = alpha_model.alpha_of(pf['secid'])
            pf['alpha_rank'] = alpha_model.alpha_of(pf['secid'] , rank = True)
        return pf.assign(name = self.name , date = date)
    
class BacktestPort(TradingPort):
    candidate_ports : ClassVar[dict[str , dict]] = Proj.Conf.TradingPort.backtest_ports
    @classmethod
    def load(cls , name : str) -> 'BacktestPort':
        if name in cls.candidate_ports:
            kwargs = {'name' : name , **cls.candidate_ports[name]} | {'backtest' : True}
            return cls(**kwargs)
        else:
            raise ValueError(f'{name} is not a valid backtest port , find configs/setting/trading_port.yaml for available backtest ports')

    @property
    def result_dir(self) -> Path:
        return PATH.rslt_trade.joinpath('backtest' , self.name)

    @property
    def portfolio_dir(self) -> Path:
        return PATH.rslt_trade.joinpath('backtest' , self.name , 'portfolio')
    
    def build(self , date : int | None = None , reset_port = False , export = True , indent : int = 1 , vb_level : Any = 2):
        date = CALENDAR.updated(date)
        df = self.build_backward(date , reset_port = reset_port , export = export , indent = indent , vb_level = vb_level)
        self.new_ports[date] = df
        return self

    def rebuild(self , date : int | None = None , export = True , indent : int = 0 , vb_level : Any = 2):
        date = CALENDAR.updated(date)
        Logger.stdout(f'Rebuild portfolio for {self.name} at {Dates(date)} start ...' , indent = indent , vb_level = vb_level)
        df = self.build_backward(date , reset_port = True , export = export , indent = indent + 1 , vb_level = vb_level)
        self.new_ports[date] = df
        return self
    
    def build_backward(self , date : int , reset_port = False , export = True , indent : int = 1 , vb_level : Any = 2) -> pd.DataFrame:
        vb_level = Proj.vb(vb_level)
        assert self.test_start > 0 , 'test_start must be positive'
        test_end = min(date , self.test_end)
        
        if test_end < self.test_start: 
            return pd.DataFrame()
        
        if reset_port:
            Logger.alert1(f'Reset {self.trading_portfolio_type.title()} Portfolio [{self.name}] for new build!' , indent = indent)
            shutil.rmtree(self.portfolio_dir , ignore_errors = True)
            self.portfolio_dir.mkdir(parents = True , exist_ok = True)
            self.result_path_account.unlink(missing_ok = True)

        date_list = CALENDAR.range(self.test_start , test_end , 'td' , step = self.step)
        date_list = date_list[date_list > self.end_date()]
        if len(date_list) == 0: 
            return pd.DataFrame()
        Logger.stdout(f'Loading alpha for {self.name} at {Dates(date_list)}' , indent = indent , vb_level = vb_level)
        alpha = self.Alpha.get(date_list)
        Logger.stdout(f'Loading universe for {self.name} at {Dates(date_list)}' , indent = indent , vb_level = vb_level)
        universe = self.Universe.get(date_list , self.exclusion)
        last_port = self.get_last_port(date_list[0])
        Logger.stdout(f'Perform portfolio building for {self.name} at {Dates(date_list)}' , indent = indent , vb_level = vb_level)
        builder = PortfolioBuilder(self.category , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , no_zone = self.no_zone , 
                                   indus_control = self.indus_control , sorting_alpha = self.sorting_alpha ,
                                   screen_ratio = self.screen_ratio , indent = indent + 1 , vb_level = vb_level + 1).setup()

        for date in date_list:
            builder.build(date)
            if export:
                df = builder.port.to_dataframe()
                self.save_port(df , date , indent = indent + 1 , vb_level = vb_level + 2)

        return pd.DataFrame()