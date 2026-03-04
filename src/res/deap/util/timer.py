import torch
from datetime import datetime

from typing import Literal, Callable
import pandas as pd

from src.proj import Logger
from src.res.deap.func import primas
from .memory import MemoryManager

class AccTimer:
    def __init__(self , key , title = '' , timer_level : Literal[1,2,3,4,5] = 3 , *, memory_check = False):
        self.key = key
        self.title = title.title()
        self.time_costs : list[float] = []
        if not title:
            self.paragraph = None
        elif timer_level != 5:
            self.paragraph = Logger.Paragraph(title , timer_level)
        else:
            self.paragraph = Logger.Timer(title , enter_vb_level = 1 , vb_level = 1)
        self.memory_check = memory_check and torch.cuda.is_available()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(key={self.key},count={self.count})'
    def __enter__(self):
        if self.paragraph is not None:
            self.paragraph.__enter__()
        self._init_time = datetime.now()
        if self.memory_check and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.gmem_start = torch.cuda.mem_get_info()[0] / MemoryManager.unit
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        time_cost = (datetime.now() - self._init_time).total_seconds()
        if exc_type is not None:
            Logger.error(f'Error in Timer {self.title or self.key}' , exc_type , exc_value)
            Logger.print_exc(exc_value)
        elif self.memory_check:
            torch.cuda.empty_cache()
            mem_end  = torch.cuda.mem_get_info()[0] / MemoryManager.unit
            Logger.success(f'Free CudaMemory {self.gmem_start:.2f}G - > {mem_end:.2f}G')

        if self.paragraph is not None:
            self.paragraph.__exit__(exc_type, exc_value, exc_traceback)
        self.time_costs.append(time_cost)
        return self
    def __bool__(self):
        return bool(self.time_costs)
    @property
    def time_cost(self) -> float:
        return sum(self.time_costs)
    @property
    def avg_time_cost(self) -> float:
        count = len(self.time_costs)
        total_time = sum(self.time_costs)
        return total_time / count if count > 0 else 0.
    @property
    def count(self) -> int:
        return len(self.time_costs)
    def __call__(self , func : Callable):
        def wrapper(*args , **kwargs):
            with self:
                return func(*args , **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper

class SilentTimer(AccTimer):
    def __init__(self , key = ''):
        self.key = key
        self.time_costs : list[float] = []
    def __enter__(self):
        self._init_time = datetime.now()
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            Logger.error(f'Error in SilentTimer {self.key}' , exc_type , exc_value)
            Logger.print_exc(exc_value)
        else:
            self.time_costs.append((datetime.now() - self._init_time).total_seconds())
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(key={self.key},count={self.count})'
    def __bool__(self):
        return bool(self.time_costs)
    @property
    def time_cost(self) -> float:
        return sum(self.time_costs)
    @property
    def avg_time_cost(self) -> float:
        count = len(self.time_costs)
        total_time = sum(self.time_costs)
        return total_time / count if count > 0 else 0.
    @property
    def count(self) -> int:
        return len(self.time_costs)


class gpTimer:
    '''
    ------------------------ gp timers ------------------------
    includes:
        PTimer     : record a process and its time cost
        AccTimer   : record a series of time costs, can average later
        EmptyTimer : do nothing
    '''
    _instance = None
    _primas_decorated = False
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , record = False) -> None:
        self.initiate(record)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'recording')

    def initiate(self , record = False) -> None:
        if self.initiated:
            return
        self.recording = record
        self.timers : dict[str, dict[str, AccTimer]] = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(timers={self.timers})'
    def __bool__(self): 
        return self.recording
    def timer(self , category : str , key : str , title : str = '' , memory_check = False , timer_level : Literal[1,2,3,4,5] = 3) -> AccTimer:
        if category not in self.timers.keys():
            self.timers[category] = {}
        if key not in self.timers[category].keys():
            self.timers[category][key] = AccTimer(key, title = title, memory_check = memory_check, timer_level = timer_level)
        return self.timers[category][key]
 
    def time_table(self):
        times : list[tuple[str , str , float , float , int]] = []
        [times.append((cat , k , v.time_cost , v.avg_time_cost , v.count)) for cat , timers in self.timers.items() for k,v in timers.items() if v]
        
        df = pd.DataFrame(times, columns=['category' , 'name', 'total_time', 'avg_time', 'count'])
        Logger.display(df , caption = 'Timer Table:')
        return df

    def decorate_primas(self):
        assert not self._primas_decorated , 'Primas already decorated'
        if not self:
            return 
        for prim_name in primas.all_prim_names():
            new_prim = self.timer(prim_name , 'prima')(getattr(primas , prim_name))
            setattr(primas , prim_name , new_prim)
        self._primas_decorated = True

    def revert_primas(self):
        if not self:
            return 
        for prim_name in primas.all_prim_names():
            setattr(primas , prim_name , primas.PrimTool.registry[prim_name])
        self._primas_decorated = False