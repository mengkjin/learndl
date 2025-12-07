import datetime , warnings
import pandas as pd
import numpy as np

from abc import ABC , abstractmethod
from matplotlib.figure import Figure
from typing import Any , Callable , Literal , Type

from src.proj import PATH , Timer
from src.func import dfs_to_excel , figs_to_pdf , display as disp
from src.data import DataBlock
from ..util import Benchmark , StockFactor

TYPE_of_TASK = Literal['optim' , 'top' , 'factor' , 't50' , 'screen' , 'revscreen']
TASK_TYPES : list[TYPE_of_TASK] = ['optim' , 'top' , 'factor' , 't50' , 'screen' , 'revscreen']

class BaseCalculator(ABC):
    TASK_TYPE : TYPE_of_TASK
    DEFAULT_BENCHMARKS : list[Benchmark|Any] | Benchmark | Any = [None]
    DEFAULT_TITLE_GROUP : str | None = None

    def __init__(self , params : dict[str,Any] | None = None , **kwargs) -> None:
        self.params : dict[str,Any] = params or {} 
        self.kwargs = kwargs
    def __repr__(self):
        return f'{self.__class__.__name__} of task {self.TASK_TYPE}(params={self.params})'
    @classmethod
    def task_name(cls): 
        return cls.__name__.lower().removeprefix(f'{cls.TASK_TYPE}_')
    
    @classmethod
    def match_name(cls , name : str):
        return name.lower() in [cls.__name__.lower() , cls.task_name()]

    class calc_manager:
        def __init__(self , *args , verbosity = 0):
            self.timer = Timer(*args , silent = verbosity < 1)
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
    def plot(self , show = False , verbosity = 0): 
        with Timer(f'    --->{self.__class__.__name__} plot' , silent = verbosity < 1):
            if self.calc_rslt.empty: 
                self.figs = {}
                return self
            try:
                figs = self.plotter()(self.calc_rslt , show = show , title_prefix = self.title_prefix)
                self.figs = {'all':figs} if isinstance(figs , Figure) else figs
            except Exception as e:
                print(f"Error when plotting {self.__class__.__name__}: {e}")
                self.figs = {}    
        return self
    @property
    def title_prefix(self) -> str:
        prefix = self.DEFAULT_TITLE_GROUP if self.DEFAULT_TITLE_GROUP else self.TASK_TYPE.title()
        if 'title_prefix' in self.kwargs:
            prefix = f'{str(self.kwargs["title_prefix"]).replace("_", " ").title()} {prefix}'
        return prefix
    
class BaseTestManager(ABC):
    TASK_TYPE : TYPE_of_TASK
    TASK_LIST : list[Type[BaseCalculator]] = []

    def __init__(self , which : str | list[str] | Literal['all'] = 'all' , project_name : str | None = None , **kwargs):
        candidates = {task.task_name():task for task in self.TASK_LIST}
            
        if which == 'all':
            self.tasks = {k:v(**kwargs) for k,v in candidates.items()}
        else:
            if isinstance(which , str): 
                which = [which]
            illegal = np.setdiff1d(which , list(candidates.keys()))
            assert len(illegal) == 0 , f'Illegal task: {illegal}'
            self.tasks = {k:v(**kwargs) for k,v in candidates.items() if k in which}
        self.kwargs = kwargs
        self.project_name = project_name
        self.get_project_name()

    def __repr__(self):
        return f'{self.__class__.__name__} of task {self.TASK_TYPE} with {len(self.tasks)} calculators'

    @classmethod
    def run_test(cls , factor : StockFactor | pd.DataFrame | DataBlock , benchmark : list[Benchmark|Any] | Any | None = 'defaults' ,
                 which = 'all' , verbosity = 2 ,**kwargs):
        pm = cls(which = which , **kwargs)
        print(f'finish pm')
        pm.calc(StockFactor(factor) , benchmark , verbosity = verbosity)
        pm.plot(show = False , verbosity = verbosity)
        return pm

    @abstractmethod
    def calc(self , factor : StockFactor , *args , verbosity = 1 , **kwargs):
        with Timer(f'{self.__class__.__name__} calc' , silent = verbosity < 1):
            for task in self.tasks.values():  
                task.calc(factor , *args , verbosity = verbosity - 1 , **kwargs) 
        return self

    def plot(self , show = False , verbosity = 1):
        with Timer(f'{self.__class__.__name__} plot' , silent = verbosity < 1):
            for task in self.tasks.values(): 
                task.plot(show = show , verbosity = verbosity - 1)
        return self

    def get_project_name(self):
        if self.project_name is None:
            start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            self.project_name = f'test_{start_time}'
        return self.project_name

    @property
    def project_path(self):
        if self.TASK_TYPE == 'optim':
            rslt_dir = PATH.rslt_optim
        elif self.TASK_TYPE == 'top':
            rslt_dir = PATH.rslt_top
        elif self.TASK_TYPE == 'factor':
            rslt_dir = PATH.rslt_factor
        elif self.TASK_TYPE == 't50':
            rslt_dir = PATH.result.joinpath('test').joinpath('t50')
        elif self.TASK_TYPE == 'screen':
            rslt_dir = PATH.result.joinpath('test').joinpath('screen')
        elif self.TASK_TYPE == 'revscreen':
            rslt_dir = PATH.result.joinpath('test').joinpath('revscreen')
        else:
            raise ValueError(f'Invalid task type: {self.TASK_TYPE}')
        rslt_dir.mkdir(parents=True , exist_ok=True)
        return rslt_dir.joinpath(self.get_project_name())

    def get_rslts(self):
        return {k:v.calc_rslt for k,v in self.tasks.items()}
    
    def get_figs(self):
        return {f'{k}@{fig_name}':fig for k,v in self.tasks.items() for fig_name , fig in v.figs.items()}
    
    def display_figs(self):
        [disp.display(fig) for fig in self.get_figs().values()]
    
    def write_down(self):
        rslts , figs = self.get_rslts() , self.get_figs()
        dfs_to_excel(rslts , self.project_path.joinpath(f'{self.TASK_TYPE}_data.xlsx') , print_prefix=f'{self.TASK_TYPE.title()} Analytic Test of {self.name_of_task()} datas')
        figs_to_pdf(figs   , self.project_path.joinpath(f'{self.TASK_TYPE}_plot.pdf')  , print_prefix=f'{self.TASK_TYPE.title()} Analytic Test of {self.name_of_task()} plots')
        return self
    
    @classmethod
    def name_of_task(cls):
        if cls.TASK_TYPE == 'optim':
            return 'Optim Portfolio'
        elif cls.TASK_TYPE == 'top':
            return 'Top Portfolio'
        elif cls.TASK_TYPE == 'factor':
            return 'Factor Performance'
        elif cls.TASK_TYPE == 't50':
            return 'Top50 Port'
        elif cls.TASK_TYPE == 'screen':
            return 'Screen Port'
        elif cls.TASK_TYPE == 'revscreen':
            return 'RevScreen Port'
        else:
            raise ValueError(f'Invalid task type: {cls.TASK_TYPE}')