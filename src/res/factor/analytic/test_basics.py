from __future__ import annotations
import warnings , re
from datetime import datetime
import pandas as pd
import numpy as np

from abc import ABC , abstractmethod
from collections.abc import Iterable
from enum import StrEnum
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any , Callable , Literal , Type

from src.proj import PATH , DB , Base , Save
from src.data import DataBlock
from ..util import Benchmark , StockFactor

def _camel_to_snake(name : str) -> str:
    """Convert CamelCase (or mixed) identifiers to lower_snake_case."""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
class TestType(StrEnum):
    FACTOR = 'factor'
    OPTIM = 'optim'
    TOP = 'top'
    T50 = 't50'
    SCREEN = 'screen'
    REVSCREEN = 'revscreen'
    REINFORCE = 'reinforce'

    @classmethod
    def values(cls) -> list[TestType]:
        return [m for m in cls]

    @classmethod
    def ensure_list(cls , x: TestType | Iterable[TestType] | Literal['all']) -> list[TestType]:
        if x == 'all':
            return list(cls)
        if isinstance(x , TestType):
            return [x]
        else:
            return [t for t in x]

    def title(self) -> str:
        if self == TestType.FACTOR:
            return self.value.title()
        else:
            return f'{self.value.title()} Port'

class CalcWarningsManager(Base.BoundLogger):
    def __init__(self , *args , indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.timer = self.logger.timer(*args , **kwargs)
    def __enter__(self):
        self.timer.__enter__()
        warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='divide by zero encountered in divide', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='invalid value encountered in multiply', category=RuntimeWarning)
    def __exit__(self , *args):
        warnings.resetwarnings()
        self.timer.__exit__(*args)

class BaseFactorAnalyticCalculator(ABC, Base.BoundLogger):
    TEST_TYPE : TestType
    DEFAULT_BENCHMARKS : list[Benchmark|Any] | Benchmark | Any = [None]

    def __init__(self , params : dict[str,Any] | None = None , **kwargs) -> None:
        super().__init__(**kwargs)
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

    def calc_manager(self):
        return CalcWarningsManager(f'{self.__class__.__name__} calc' , indent = self.indent , vb_level = self.vb_level)

    @abstractmethod
    def calculator(self) -> Callable[...,pd.DataFrame]: 
        """Define calculator"""
    @abstractmethod
    def plotter(self) -> Callable: 
        """Define plotter"""
    @abstractmethod
    def calc(self , *args , **kwargs):
        self.calc_rslt = self.calculator()(*args , **kwargs)
        return self
    def plot(self , show = False): 
        with self.logger.timer('plot'):
            if self.calc_rslt.empty: 
                self.figs = {}
                return self
            try:
                figs = self.plotter()(self.calc_rslt , show = show , title_prefix = self.title_prefix)
                self.figs = {'all':figs} if isinstance(figs , Figure) else figs
            except Exception as e:
                self.logger.error(f"Error when plotting {self.__class__.__name__}: {e}")
                self.logger.print_exc(e)
                self.figs = {}    
        return self
    @property
    def title_prefix(self) -> str:
        if title_prefix := self.kwargs.get('title_prefix' , None):
            return f'{title_prefix} {self.TEST_TYPE.title()}'
        return self.TEST_TYPE.title()
        
class BaseFactorAnalyticTest(ABC, Base.BoundLogger):
    TEST_TYPE : TestType
    TASK_LIST : list[Type[BaseFactorAnalyticCalculator]] = []

    def __init__(
        self , test_path : Base.types.strPath | None = None , 
        resume : bool = False, save_resumable : bool = False , start : int = -1 , end : int = 99991231 , 
        which : str | list[str] | Literal['all'] = 'all' , **kwargs
    ):
        super().__init__(**kwargs)
        candidates = {task.task_name():task for task in self.TASK_LIST}
        self.create_time = datetime.now()
        
        self.test_path = test_path
        self.resume = resume
        self.save_resumable = save_resumable
        self.start = start
        self.end = end
        self.kwargs = kwargs
        vb_kwgs = {'indent':self.indent + 1 , 'vb_level':self.vb_level + 2}
        if which == 'all':
            self.tasks = {k:v(**(self.kwargs | vb_kwgs)) for k,v in candidates.items()}
        else:
            if isinstance(which , str): 
                which = [which]
            illegal = np.setdiff1d(which , list(candidates.keys()))
            assert len(illegal) == 0 , f'Illegal task: {illegal}'
            self.tasks = {k:v(**self.kwargs | vb_kwgs) for k,v in candidates.items() if k in which}

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @classmethod
    def create(cls , test_path : Base.types.strPath | None = None , resume : bool = False , save_resumable : bool = False , 
               start : int = -1 , end : int = 99991231 , which = 'all' , **kwargs):
        testor = cls(test_path , resume , save_resumable , start , end , which , **kwargs)
        return testor

    def proceed(self , factor : StockFactor | pd.DataFrame | DataBlock , benchmark : list[Benchmark|Any] | Any | None = 'defaults' , **kwargs):
        self.calc(StockFactor(factor) , benchmark)
        self.plot(show = False)
        return self

    def calc(self , *args , **kwargs):
        with self.logger.timer(f'{self.class_snake_name}.calc' , enter_vb = 2 , add_prefix = False):
            for task in self.tasks.values():  
                task.calc(*args , **kwargs) 
        return self

    def plot(self , show = False):
        with self.logger.timer(f'{self.class_snake_name}.plot' , enter_vb = 2 , add_prefix = False):
            for task in self.tasks.values(): 
                task.plot(show = show)
        return self

    @property
    def class_snake_name(self) -> str:
        return _camel_to_snake(self.__class__.__name__)

    @property
    def test_name(self) -> str:
        return f'{self.class_snake_name}_{self.create_time.strftime('%Y%m%d%H%M%S')}'

    @property
    def test_path(self):
        if self._test_path is not None:
            return self._test_path
        else:
            rslt_dir = PATH.rslt_factor.joinpath(str(self.TEST_TYPE))
            rslt_dir.mkdir(parents=True , exist_ok=True)
            return rslt_dir.joinpath(self.test_name)

    @test_path.setter
    def test_path(self , path : Base.types.strPath | None):
        self._test_path = path if path is None else Path(path)

    @property
    def resume_path(self) -> Path | None:
        if self._test_path is None:
            return None
        else:
            return self._test_path.joinpath(_camel_to_snake(self.__class__.__name__))

    @classmethod
    def last_portfolio_date(cls , test_path : Base.types.strPath | None = None):
        if test_path is None:
            return 19000101
        else:
            portfolio_resume_path = Path(test_path) / _camel_to_snake(cls.__name__) / 'portfolio'
            last_dates = [DB.load_df_max_date(path) for path in portfolio_resume_path.glob('*.feather')]
            return min(last_dates) if len(last_dates) else 19000101

    @property
    def factor_stats_resume_path(self):
        assert self.TEST_TYPE == TestType.FACTOR , self.TEST_TYPE
        if self.resume_path is None:
            return None
        else:
            return self.resume_path.joinpath(f'factor_stats')

    @classmethod
    def factor_stats_saved_dates(cls , test_path : Base.types.strPath | None = None) -> np.ndarray:
        if test_path is None:
            return np.array([] , dtype=int)
        else:
            from src.res.factor.util.classes.stock_factor import CacheFactorStats
            stats_path = Path(test_path) / _camel_to_snake(cls.__name__) / 'factor_stats'
            return CacheFactorStats.saved_dates(stats_path)

    def get_rslts(self) -> dict[str, pd.DataFrame]:
        return {k:v.calc_rslt for k,v in self.tasks.items()}
    
    def get_figs(self):
        return {f'{k}@{fig_name}':fig for k,v in self.tasks.items() for fig_name , fig in v.figs.items()}
    
    def display_figs(self):
        [self.logger.display(fig) for fig in self.get_figs().values()]
    
    def write_down(self):
        rslts , figs = self.get_rslts() , self.get_figs()
        Save.dfs(
            rslts , self.test_path.joinpath(f'{self.TEST_TYPE}_data.xlsx') , async_save = True ,
            prefix=f'{self.__class__.__name__} Analytic Datas' , indent = self.indent + 1 , vb_level = self.vb_level + 1)
        Save.figs(
            figs , self.test_path.joinpath(f'{self.TEST_TYPE}_plot.pdf')  , async_save = True ,
            prefix=f'{self.__class__.__name__} Analytic Plots' , indent = self.indent + 1 , vb_level = self.vb_level + 1)
        return self

    def save(self , path : Base.types.strPath):
        """save intermediate data to path for future use"""
        ...

    def load(self , path : Base.types.strPath):
        """load intermediate data from path for future use"""

    @classmethod
    def get_test_class(cls , test_type : TestType):
        """
        get the test class for the given test type
        """
        match test_type:
            case TestType.FACTOR:
                from src.res.factor.analytic.factor_perf import FactorPerfTest
                return FactorPerfTest
            case TestType.OPTIM:
                from src.res.factor.analytic.fmp_optim import OptimFMPTest
                return OptimFMPTest
            case TestType.TOP:
                from src.res.factor.analytic.fmp_top import TopFMPTest
                return TopFMPTest
            case TestType.T50:
                from src.res.factor.analytic.fmp_t50 import T50FMPTest
                return T50FMPTest
            case TestType.SCREEN:
                from src.res.factor.analytic.fmp_screen import ScreenFMPTest
                return ScreenFMPTest
            case TestType.REVSCREEN:
                from src.res.factor.analytic.fmp_revscreen import RevScreenFMPTest
                return RevScreenFMPTest
            case TestType.REINFORCE:
                from src.res.factor.analytic.fmp_reinforce import ReinforceFMPTest
                return ReinforceFMPTest
            case _:
                raise ValueError(f'Invalid test type: {test_type}')