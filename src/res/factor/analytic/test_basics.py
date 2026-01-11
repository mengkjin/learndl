import warnings
from datetime import datetime
import pandas as pd
import numpy as np

from abc import ABC , abstractmethod
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any , Callable , Literal , Type

from src.proj import PATH , Logger , DB
from src.proj.func import dfs_to_excel , figs_to_pdf , camel_to_snake
from src.data import DataBlock
from ..util import Benchmark , StockFactor

TYPE_of_TEST = Literal['factor' , 'optim' , 'top' , 't50' , 'screen' , 'revscreen']
TEST_TYPES : list[TYPE_of_TEST] = ['factor' , 'optim' , 'top' , 't50' , 'screen' , 'revscreen']

def test_title(test_type : str) -> str:
    assert test_type in TEST_TYPES , f'Invalid test type: {test_type}'
    if test_type == 'factor':
        return test_type.title()
    else:
        return f'{test_type.title()} Port'

class TestTitle:
    def __get__(self,instance,owner) -> str:
        return test_title(str(getattr(owner, 'TEST_TYPE')))

class BaseFactorAnalyticCalculator(ABC):
    TEST_TYPE : TYPE_of_TEST
    DEFAULT_BENCHMARKS : list[Benchmark|Any] | Benchmark | Any = [None]
    TEST_TITLE = TestTitle()

    def __init__(self , params : dict[str,Any] | None = None , **kwargs) -> None:
        self.params : dict[str,Any] = params or {} 
        self.kwargs = kwargs
    def __repr__(self):
        return f'{self.__class__.__name__} of task {self.TEST_TYPE}(params={self.params})'
    @classmethod
    def task_name(cls): 
        return cls.__name__.lower().removeprefix(f'{cls.TEST_TYPE}_')
    
    @classmethod
    def match_name(cls , name : str):
        return name.lower() in [cls.__name__.lower() , cls.task_name()]

    class calc_manager:
        def __init__(self , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
            self.timer = Logger.Timer(*args , indent = indent , vb_level = vb_level , **kwargs)
        def __enter__(self):
            self.timer.__enter__()
            warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='divide by zero encountered in divide', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='invalid value encountered in multiply', category=RuntimeWarning)
        def __exit__(self , *args):
            warnings.resetwarnings()
            self.timer.__exit__(*args)

    @abstractmethod
    def calculator(self) -> Callable[...,pd.DataFrame]: '''Define calculator'''
    @abstractmethod
    def plotter(self) -> Callable: '''Define plotter'''
    @abstractmethod
    def calc(self , *args , **kwargs):
        self.calc_rslt = self.calculator()(*args , **kwargs)
        return self
    def plot(self , show = False , indent : int = 0 , vb_level : int = 1): 
        with Logger.Timer(f'{self.__class__.__name__} plot' , indent = indent , vb_level = vb_level):
            if self.calc_rslt.empty: 
                self.figs = {}
                return self
            try:
                figs = self.plotter()(self.calc_rslt , show = show , title_prefix = self.title_prefix)
                self.figs = {'all':figs} if isinstance(figs , Figure) else figs
            except Exception as e:
                Logger.error(f"Error when plotting {self.__class__.__name__}: {e}")
                self.figs = {}    
        return self
    @property
    def title_prefix(self) -> str:
        if title_prefix := self.kwargs.get('title_prefix' , None):
            return f'{title_prefix} {self.TEST_TITLE}'
        return self.TEST_TITLE
    
class BaseFactorAnalyticTest(ABC):
    TEST_TYPE : TYPE_of_TEST
    TASK_LIST : list[Type[BaseFactorAnalyticCalculator]] = []
    TEST_TITLE = TestTitle()

    def __init__(
        self , test_path : Path | str | None = None , 
        resume : bool = False, save_resumable : bool = False , start_dt : int = -1 , end_dt : int = 99991231 , 
        which : str | list[str] | Literal['all'] = 'all' , **kwargs
    ):
        candidates = {task.task_name():task for task in self.TASK_LIST}
        self.create_time = datetime.now()
        self.kwargs = kwargs
        self.test_path = test_path
        self.resume = resume
        self.save_resumable = save_resumable
        self.start_dt = start_dt
        self.end_dt = end_dt
        if which == 'all':
            self.tasks = {k:v(**kwargs) for k,v in candidates.items()}
        else:
            if isinstance(which , str): 
                which = [which]
            illegal = np.setdiff1d(which , list(candidates.keys()))
            assert len(illegal) == 0 , f'Illegal task: {illegal}'
            self.tasks = {k:v(**kwargs) for k,v in candidates.items() if k in which}

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @classmethod
    def run_test(cls , factor : StockFactor | pd.DataFrame | DataBlock , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
                test_path : Path | str | None = None , resume : bool = False , save_resumable : bool = False , 
                 indent : int = 0 , vb_level : int = 1 , start_dt : int = -1 , end_dt : int = 99991231 , which = 'all' , **kwargs):
        pm = cls(test_path , resume , save_resumable , start_dt , end_dt , which , **kwargs)
        pm.calc(StockFactor(factor) , benchmark , indent = indent , vb_level = vb_level)
        pm.plot(show = False , indent = indent , vb_level = vb_level)
        return pm

    def calc(self , *args , indent : int = 0 , vb_level : int = 1 , **kwargs):
        with Logger.Timer(f'{self.__class__.__name__}.calc' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 2):
            for task in self.tasks.values():  
                task.calc(*args , indent = indent + 1 , vb_level = vb_level + 2 , **kwargs) 
        return self

    def plot(self , show = False , indent : int = 0 , vb_level : int = 1):
        with Logger.Timer(f'{self.__class__.__name__}.plot' , indent = indent , vb_level = vb_level , enter_vb_level = vb_level + 2):
            for task in self.tasks.values(): 
                task.plot(show = show , indent = indent + 1 , vb_level = vb_level + 2)
        return self

    @property
    def class_snake_name(self) -> str:
        return camel_to_snake(self.__class__.__name__)

    @property
    def test_name(self) -> str:
        return f'{self.class_snake_name}_{self.create_time.strftime('%Y%m%d%H%M%S')}'

    @property
    def test_path(self):
        if self._test_path is not None:
            return self._test_path
        else:
            rslt_dir = PATH.rslt_test.joinpath(self.TEST_TYPE)
            rslt_dir.mkdir(parents=True , exist_ok=True)
            return rslt_dir.joinpath(self.test_name)

    @test_path.setter
    def test_path(self , path : Path | str | None):
        self._test_path = path if path is None else Path(path)

    @property
    def resume_path(self) -> Path | None:
        if self._test_path is None:
            return None
        else:
            return self._test_path.joinpath(camel_to_snake(self.__class__.__name__))

    @classmethod
    def last_portfolio_date(cls , test_path : Path | str | None = None):
        if test_path is None:
            return 19000101
        else:
            portfolio_resume_path = Path(test_path) / camel_to_snake(cls.__name__) / 'portfolio'
            last_dates = [DB.load_df_max_date(path) for path in portfolio_resume_path.glob('*.feather')]
            return min(last_dates) if len(last_dates) else 19000101

    @property
    def factor_stats_resume_path(self):
        assert self.TEST_TYPE in ['factor'] , self.TEST_TYPE
        if self.resume_path is None:
            return None
        else:
            return self.resume_path.joinpath(f'factor_stats')

    def get_rslts(self):
        return {k:v.calc_rslt for k,v in self.tasks.items()}
    
    def get_figs(self):
        return {f'{k}@{fig_name}':fig for k,v in self.tasks.items() for fig_name , fig in v.figs.items()}
    
    def display_figs(self):
        [Logger.Display(fig) for fig in self.get_figs().values()]
    
    def write_down(self):
        rslts , figs = self.get_rslts() , self.get_figs()
        dfs_to_excel(rslts , self.test_path.joinpath(f'{self.TEST_TYPE}_data.xlsx') , print_prefix=f'{self.__class__.__name__} Analytic Datas')
        figs_to_pdf(figs   , self.test_path.joinpath(f'{self.TEST_TYPE}_plot.pdf')  , print_prefix=f'{self.__class__.__name__} Analytic Plots')
        return self

    def save(self , path : Path | str):
        """save intermediate data to path for future use"""
        ...

    def load(self , path : Path | str):
        """load intermediate data from path for future use"""