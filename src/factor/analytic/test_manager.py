import datetime , warnings
import pandas as pd
import numpy as np

from abc import ABC , abstractmethod
from matplotlib.figure import Figure
from typing import Any , Callable , Literal , Type

from src.basic import PATH
from src.func import dfs_to_excel , figs_to_pdf
from src.func import display as disp
from src.data import DataBlock
from src.factor.util import Benchmark , StockFactor

TASK_TYPES = ['optim' , 'top' , 'factor']
TYPE_of_TASK = Literal['optim' , 'top' , 'factor']

class BaseCalculator(ABC):
    TASK_TYPE : TYPE_of_TASK
    DEFAULT_BENCHMARKS : list[Benchmark|Any] | Benchmark | Any = [None]

    def __init__(self , **kwargs) -> None:
        self.params : dict[str,Any] = kwargs
    def __repr__(self):
        return f'{self.__class__.__name__} of task {self.TASK_TYPE}(params={self.params})'
    @classmethod
    def match_name(cls , name : str):
        candidate_names = [
            cls.__name__.lower() ,
            cls.__name__.lower().removeprefix(f'{cls.TASK_TYPE}_'),
        ]
        return name.lower() in candidate_names

    class suppress_warnings:
        def __enter__(self):
            warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='divide by zero encountered in divide', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='invalid value encountered in multiply', category=RuntimeWarning)
        def __exit__(self , *args):
            warnings.resetwarnings()

    @classmethod
    def task_name(cls): 
        return cls.__name__.lower().removeprefix(f'{cls.TASK_TYPE}_')
    @abstractmethod
    def calculator(self) -> Callable[...,pd.DataFrame]: '''Define calculator'''
    @abstractmethod
    def plotter(self) -> Callable: '''Define plotter'''
    @abstractmethod
    def calc(self , *args , **kwargs):
        self.calc_rslt = self.calculator()(*args , **kwargs)
        return self
    def plot(self , show = False , verbosity = 0): 
        if self.calc_rslt.empty: 
            self.figs = {}
            return self
        figs = self.plotter()(self.calc_rslt , show = show)
        self.figs = {'all':figs} if isinstance(figs , Figure) else figs
        if verbosity > 0: print(f'    --->{self.__class__.__name__} plot Finished!')
        return self
    
class BaseTestManager(ABC):
    TASK_TYPE : TYPE_of_TASK
    TASK_LIST : list[Type[BaseCalculator]] = []

    def __init__(self , which : str | list[str] | Literal['all'] = 'all' , project_name : str | None = None , **kwargs):
        candidates = {task.task_name():task for task in self.TASK_LIST}
        if which == 'all':
            self.tasks = {k:v(**kwargs) for k,v in candidates.items()}
        else:
            if isinstance(which , str): which = [which]
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
        pm.calc(StockFactor(factor) , benchmark , verbosity = verbosity)
        pm.plot(show = False , verbosity = verbosity)
        return pm

    @abstractmethod
    def calc(self , factor : StockFactor , *args , verbosity = 1 , **kwargs):
        for task in self.tasks.values():  task.calc(factor , *args , verbosity = verbosity - 1 , **kwargs) 
        if verbosity > 0: print(f'{self.__class__.__name__} calc Finished!')
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
        return rslt_dir.joinpath(self.get_project_name())

    def plot(self , show = False , verbosity = 1):
        for task in self.tasks.values(): task.plot(show = show , verbosity = verbosity - 1)
        if verbosity > 0: print(f'{self.__class__.__name__} plot Finished!')
        return self

    def get_rslts(self):
        return {k:v.calc_rslt for k,v in self.tasks.items()}
    
    def get_figs(self):
        return {f'{k}@{fig_name}':fig for k,v in self.tasks.items() for fig_name , fig in v.figs.items()}
    
    def display_figs(self):
        [disp.display(fig) for fig in self.get_figs().values()]
    
    def write_down(self):
        rslts , figs = self.get_rslts() , self.get_figs()
        dfs_to_excel(rslts , self.project_path.joinpath('data.xlsx') , print_prefix=f'Analytic Test of {self.name_of_task()} datas')
        figs_to_pdf(figs   , self.project_path.joinpath('plot.pdf')  , print_prefix=f'Analytic Test of {self.name_of_task()} plots')
        return self
    
    @classmethod
    def name_of_task(cls):
        if cls.TASK_TYPE == 'optim':
            return 'Optim Portfolio'
        elif cls.TASK_TYPE == 'top':
            return 'Top Portfolio'
        elif cls.TASK_TYPE == 'factor':
            return 'Factor Performance'