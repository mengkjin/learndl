import torch
from datetime import datetime
from src.proj import Logger
import pandas as pd

from .memory import MemoryManager

class gpTimer:
    '''
    ------------------------ gp timers ------------------------
    includes:
        PTimer     : record a process and its time cost
        AccTimer   : record a series of time costs, can average later
        EmptyTimer : do nothing
    '''
    def __init__(self , record = False) -> None:
        self.recording = record
        self.recorder = {}
        self.df_cols = {}

    class PTimer:
        def __init__(self , key , record = False , target_dict : dict | None = None , printing = True , print_str = None , memory_check = False):
            self.key = key
            self.record = record
            self.target_dict = target_dict or {}
            self.printing = printing
            self.print_str = key if print_str is None else print_str
            self.memory_check = memory_check and torch.cuda.is_available()
        def __enter__(self):
            if self.printing: 
                Logger.stdout('-' * 20 + f' {self.key} ' + '-' * 20)
            self._init_time = datetime.now()
            if self.memory_check and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.gmem_start = torch.cuda.mem_get_info()[0] / MemoryManager.unit
            return self
        def __exit__(self, exc_type, exc_value, exc_traceback):
            if exc_type is not None:
                Logger.error(f'Error in PTimer {self.key}' , exc_type , exc_value)
                Logger.print_exc(exc_value)
            else:
                time_cost = (datetime.now() - self._init_time).total_seconds()
                if self.memory_check:
                    torch.cuda.empty_cache()
                    mem_end  = torch.cuda.mem_get_info()[0] / MemoryManager.unit
                    mem_info = f', Free CudaMemory {self.gmem_start:.2f}G - > {mem_end:.2f}G' 
                else:
                    mem_info = ''
                if self.record: 
                    self.append_time(self.target_dict , self.key , time_cost)
                if self.printing: 
                    Logger.success(f'{self.print_str} , Cost {time_cost:.2f} Secs' + mem_info)
                return self
        def add_string(self , new_str):
            self.print_str = self.print_str + new_str
        @staticmethod
        def append_time(target_dict , key , time_cost):
            if key not in target_dict.keys():
                target_dict[key] = [time_cost]
            else:
                target_dict[key].append(time_cost)

    class AccTimer:
        def __init__(self , key = ''):
            self.key   = key
            self.clear()
        def __enter__(self):
            self._init_time = datetime.now()
        def __exit__(self, exc_type, exc_value, exc_traceback):
            if exc_type is not None:
                Logger.error(f'Error in AccTimer {self.key}' , exc_type , exc_value)
                Logger.print_exc(exc_value)
            else:
                self.time  += (datetime.now() - self._init_time).total_seconds()
                self.count += 1
        def __repr__(self) -> str:
            return f'time : {self.time} , count {self.count}'
        def avgtime(self , pop_out = False):
            avg = self.time if self.count == 0 else self.time / self.count
            if pop_out: 
                self.clear()
            return avg
        def clear(self):
            self.time  = 0.
            self.count = 0
        
    def __call__(self , key , printing = True , df_cols = True , print_str = None , memory_check = False):
        if df_cols: 
            self.df_cols.update({key:True})
        return self.PTimer(key , self.recording , self.recorder , printing = printing , print_str = print_str , memory_check = memory_check)
    def __repr__(self):
        return self.recorder.__repr__()
    def __bool__(self): 
        return True
    def acc_timer(self , key):
        if key not in self.recorder.keys(): 
            self.recorder[key] = self.AccTimer(key)
        assert isinstance(self.recorder[key] , self.AccTimer) , self.recorder[key]
        return self.recorder[key]
    def append_time(self , key , time_cost , df_cols = True):
        if df_cols: 
            self.df_cols.update({key:True})
        self.PTimer.append_time(self.recorder , key , time_cost)
    def time_table(self , columns = None , showoff = False , dtype = float):
        if columns is None:
            df = pd.DataFrame(data = {k:self.recorder[k] for k,v in self.df_cols.items() if v and k in self.recorder.keys()} , dtype=dtype) 
        else:
            df = pd.DataFrame(data = {k:self.recorder[k] for k in columns} , dtype=dtype) 
        if showoff: 
            with pd.option_context('display.width' , 160 ,  'display.max_colwidth', 10 , 'display.precision', 4,):
                Logger.stdout(df)
        return df.round(6)
