import time
import pandas as pd
import numpy as np

from typing import Callable
from .silence import SILENT

class Duration:
    def __init__(self , duration : int | float):
        assert duration >= 0 , f"duration must be a positive duration , but got {duration}"
        self.duration = duration
    def __str__(self):
        if self.duration < 60:
            return f"{self.duration:.2f} Secs"
        elif self.duration < 3600:
            return f"{int(self.duration / 60)} Min {int(self.duration % 60)} Secs"
        else:
            return f"{int(self.duration / 3600)} Hr {int(self.duration % 3600 / 60)} Min {int(self.duration % 60)} Secs"
    def __repr__(self):
        return f"Duration(duration={self.duration})"
    @property
    def hours(self):
        return self.duration / 3600
    @property
    def minutes(self):
        return self.duration / 60
    @property
    def seconds(self):
        return self.duration
    @property
    def days(self):
        return self.duration / 86400
    @property
    def fmtstr(self):
        # Calculate time components
        
        # Store components in a dictionary for f-string formatting
        if self.duration < 1:
            return '<1 Second'
        elif self.duration < 60:
            return f'{self.duration:.1f} Secs'
        else:
            days, remainder = divmod(self.duration, 86400) # 86400 seconds in a day
            hours, remainder = divmod(remainder, 3600)    # 3600 seconds in an hour
            minutes, seconds = divmod(remainder, 60)      # 60 seconds in a minute
        
            fmtstrs = []
            if days > 0:
                fmtstrs.append(f'{days:.0f} Day')
            if hours >= 1:
                fmtstrs.append(f'{hours:.0f} Hour')
            if minutes >= 1:
                fmtstrs.append(f'{minutes:.0f} Min')
            if seconds >= 1:
                fmtstrs.append(f'{seconds:.0f} Sec')
            return ' '.join(fmtstrs)
    
class Timer:
    '''simple timer to print out time'''
    def __init__(self , *args , newline = False , exit_only = True , silent = False): 
        self.newline = newline
        self.exit_only = exit_only
        self.silent = silent
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        if not self.silent and not SILENT and not self.exit_only: 
            print(self._str_at_first() , end='\n' if self.newline else '')
    def __exit__(self, type, value, trace):
        if not self.silent and not SILENT:
            print(self._str_at_exit())
    def _str_at_first(self):
        return f'{self.key} start ... '
    def _str_at_exit(self):
        time_cost = time.time() - self.start_time
        if time_cost < 1000:
            text = f'finished! Cost {time_cost:.2f} secs'
        elif time_cost < 3600:
            minutes, seconds = divmod(time_cost, 60)
            text =  f'finished! Cost {minutes:.0f} mins {seconds:.1f} secs'
        else:
            hours, remainder = divmod(time_cost, 3600)
            minutes , seconds = divmod(remainder, 60)
            text = f'finished! Cost {hours:.0f} hours {minutes:.0f} minutes {seconds:.1f} seconds'

        if self.exit_only:
            return f'{self.key} {text}'
        elif self.newline:
            return text

class PTimer:
    '''process timer , call to record and .summarize() to print out summary'''
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {} if record else None
    class ptimer:
        def __init__(self , target : dict[str,list[float]] | None , key):
            self.target , self.key = target , key
            if self.target is not None and key not in self.target.keys(): 
                self.target[self.key] = []
        def __enter__(self):
            if self.target is not None: 
                self.start_time = time.time()
        def __exit__(self, type, value, trace):
            if self.target is not None: 
                self.target[self.key].append(time.time() - self.start_time)

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
                                columns = pd.Index(['keys' , 'num_calls', 'total_time']))
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
        if not SILENT: 
            self.printer(f'{self.name} Finished! Cost {self.time_str(time.time()-self.start_time)}')
    @staticmethod
    def time_str(seconds : float | int):
        time_str = ''
        if (hours := seconds // 3600) > 0: 
            time_str += f'{hours:.0f} Hours '
        if (minutes := (seconds - hours * 3600) // 60) > 0: 
            time_str += f'{minutes:.0f} Minutes '
        time_str += f'{seconds - hours * 3600 - minutes * 60:.1f} Seconds'
        return time_str

