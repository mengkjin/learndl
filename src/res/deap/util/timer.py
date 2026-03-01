import torch
from datetime import datetime
from src.proj import Logger
import pandas as pd
import numpy as np
from .memory import MemoryManager

class Timer:
    def __init__(self , key , memory_check = False, print_str = ''):
        self.key = key
        self.print_str = print_str
        self.memory_check = memory_check and torch.cuda.is_available()
        self.time_costs : list[float] = []
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(key={self.key},count={self.count})'
    def __enter__(self):
        print_str = self.print_str or self.key
        Logger.stdout('-' * 20 + f' {print_str} ' + '-' * 20)
        self._init_time = datetime.now()
        if self.memory_check and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.gmem_start = torch.cuda.mem_get_info()[0] / MemoryManager.unit
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        time_cost = (datetime.now() - self._init_time).total_seconds()
        print_str = self.print_str or self.key
        if exc_type is not None:
            Logger.error(f'Error in PTimer {print_str}' , exc_type , exc_value)
            Logger.print_exc(exc_value)
        else:
            if self.memory_check:
                torch.cuda.empty_cache()
                mem_end  = torch.cuda.mem_get_info()[0] / MemoryManager.unit
                mem_info = f', Free CudaMemory {self.gmem_start:.2f}G - > {mem_end:.2f}G' 
            else:
                mem_info = ''

            Logger.success(f'{print_str} Finished! Cost {time_cost:.2f} Secs' + mem_info)
        self.time_costs.append(time_cost)
        return self
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

class SilentTimer:
    def __init__(self , key = ''):
        self.key = key
        self.time_costs : list[float] = []
    def __enter__(self):
        self._init_time = datetime.now()
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            Logger.error(f'Error in AccTimer {self.key}' , exc_type , exc_value)
            Logger.print_exc(exc_value)
        else:
            self.time_costs.append((datetime.now() - self._init_time).total_seconds())
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(key={self.key},count={self.count})'
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
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , record = False) -> None:
        self.initialize(record)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'recording')

    def initialize(self , record = False) -> None:
        if self.initiated:
            return
        self.recording = record
        self.timers : dict[str, SilentTimer] = {}
        self.silent_timers : dict[str, Timer] = {}
        self.other_timers : dict[str, list[float]] = {}

    def enable(self):
        self.recording = True
        return self

    def disable(self):
        self.recording = False
        return self

    def __call__(self , key , memory_check = False):
        timer = Timer(key , memory_check = memory_check)
        self.silent_timers[key] = timer
        return timer
    def __repr__(self):
        return {self.__class__.__name__:self.silent_timers.keys(),
                'timers_acc':self.timers.keys(),
                'timer_recorder':self.other_timers.keys()}
    def __bool__(self): 
        return True
    def timer(self , key , print_str = '' , memory_check = False) -> Timer:
        if key not in self.silent_timers.keys():
            self.silent_timers[key] = Timer(key, print_str = print_str, memory_check = memory_check)
        return self.silent_timers[key]
    def silent_timer(self , key) -> SilentTimer:
        if key not in self.timers.keys():
            self.timers[key] = SilentTimer(key)
        return self.timers[key]
        
    def append_time(self , key , time_cost):
        if key not in self.other_timers.keys():
            self.other_timers[key] = []
        self.other_timers[key].append(time_cost)
    def time_table(self):
        all_times : list[tuple[str , float , float , int]] = []
        for k,v in self.other_timers.items():
            all_times.append((k , np.sum(v).item() , np.mean(v).item() , len(v)))
        for k,v in self.silent_timers.items():
            all_times.append((k , v.time_cost , v.avg_time_cost , v.count))
        for k,v in self.timers.items():
            all_times.append((k , v.time_cost , v.avg_time_cost , v.count))

        df = pd.DataFrame(all_times, columns=['name', 'total_time', 'avg_time', 'count'])
        with pd.option_context('display.width' , 200 ,  'display.max_colwidth', 20 , 'display.precision', 4,):
            Logger.stdout(df)
        return df
