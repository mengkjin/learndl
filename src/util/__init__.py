import time
import numpy as np
import pandas as pd

from typing import Callable

from . import (
    buffer , callback , config , device , ensemble , loader , logger , metric , optim , store
)
from .buffer import BufferSpace
from .callback import CallBackManager
from .config import TrainConfig
from .device import Device
from .ensemble import EnsembleModels
from .loader import DataloaderStored , LoaderWrapper
from .logger import Logger
from .metric import Metrics , MetricsAggregator
from .optim import Optimizer
from .store import Checkpoint , Deposition , Storage

class Filtered:
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