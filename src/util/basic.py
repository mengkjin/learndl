import torch
import numpy as np
import pandas as pd
import gc , time , os , psutil

from copy import deepcopy

class Timer:
    def __init__(self , *args):
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        print(self.key , '...', end='')
    def __exit__(self, type, value, trace):
        print(f'... cost {time.time()-self.start_time:.2f} secs')

class ProcessTimer:
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {} if record else None

    class ptimer:
        def __init__(self , target_dict = None , *args):
            self.target_dict = target_dict
            if self.target_dict is not None:
                self.key = '/'.join(args)
                if self.key not in self.target_dict.keys():
                    self.target_dict[self.key] = []
        def __enter__(self):
            if self.target_dict is not None:
                self.start_time = time.time()
        def __exit__(self, type, value, trace):
            if self.target_dict is not None:
                time_cost = time.time() - self.start_time
                self.target_dict[self.key].append(time_cost)

    def __call__(self , *args):
        return self.ptimer(self.recorder , *args)
    
    def print(self):
        if self.recorder is not None:
            keys = list(self.recorder.keys())
            num_calls = [len(self.recorder[k]) for k in keys]
            total_time = [np.sum(self.recorder[k]) for k in keys]
            tb = pd.DataFrame({'keys':keys , 'num_calls': num_calls, 'total_time': total_time})
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))
            
class MemoryPrinter:
    def __init__(self) -> None:
        pass
    def __repr__(self) -> str:
        return 'Used: {:.2f}G; Free {:.2f}G'.format(
            float(psutil.virtual_memory().used)/1024**3,
            float(psutil.virtual_memory().free)/1024**3)
    def print(self):
        print(self.__repr__())
        
class FilteredIterator:
    def __init__(self, iterable, condition):
        self.iterable  = iter(iterable)
        self.condition = condition if callable(condition) else iter(condition)
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item
        