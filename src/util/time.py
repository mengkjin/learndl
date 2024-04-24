import numpy as np
import pandas as pd
import time

from typing import Callable

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
            keys = list(self.recorder.keys())
            num_calls = [len(self.recorder[k]) for k in keys]
            total_time = [np.sum(self.recorder[k]) for k in keys]
            tb = pd.DataFrame({'keys':keys , 'num_calls': num_calls, 'total_time': total_time})
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))
        
class Timer:
    '''simple timer to print out time'''
    def __init__(self , *args): self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        print(self.key , '...', end='')
    def __exit__(self, type, value, trace): print(f'... cost {time.time()-self.start_time:.2f} secs')
        