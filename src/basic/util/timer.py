import time
import pandas as pd
import numpy as np

from typing import Callable
from .silence import SILENT

class Timer:
    '''simple timer to print out time'''
    def __init__(self , *args , newline = False , exit_only = True): 
        self.newline = newline
        self.exit_only = exit_only
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        if not SILENT and not self.exit_only: 
            print(self.key , end=' start!\n' if self.newline else '...')
    def __exit__(self, type, value, trace):
        if not SILENT:
            print(self.key if self.newline or not self.exit_only else '...' , f'finished! Cost {time.time()-self.start_time:.2f} secs')

class PTimer:
    '''process timer , call to record and .summarize() to print out summary'''
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {} if record else None
    class ptimer:
        def __init__(self , target : dict[str,list[float]] | None , key):
            self.target , self.key = target , key
            if self.target is not None and key not in self.target.keys(): self.target[self.key] = []
        def __enter__(self):
            if self.target is not None: self.start_time = time.time()
        def __exit__(self, type, value, trace):
            if self.target is not None: self.target[self.key].append(time.time() - self.start_time)

    def func_timer(self , func : Callable):
        def wrapper(*args , **kwargs):
            with self.ptimer(self.recorder , func.__name__):
                return func(*args , **kwargs)
        return wrapper if self.recording else func

    def __call__(self , *args):
        return self.ptimer(self.recorder , '/'.join(args))
    
    def summarize(self):
        if self.recorder is not None:
            tb = pd.DataFrame([[k , len(self.recorder[k]) , np.sum(self.recorder[k])] for k in self.recorder.keys()] ,
                                columns = ['keys' , 'num_calls', 'total_time'])
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))

class BigTimer:
    '''big timer to print out time'''
    def __init__(self , printer = print , name = None):
        self.printer = printer
        self.name = name if name else 'Whole Process'
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, *args): 
        if not SILENT: self.printer(f'{self.name} Finished! Cost {self.time_str(time.time()-self.start_time)}')
    @staticmethod
    def time_str(seconds : float | int):
        time_str = ''
        if (hours := seconds // 3600) > 0: time_str += f'{hours:.0f} Hours '
        if (minutes := (seconds - hours * 3600) // 60) > 0: time_str += f'{minutes:.0f} Minutes '
        time_str += f'{seconds - hours * 3600 - minutes * 60:.1f} Seconds'
        return time_str

