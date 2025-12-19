import pandas as pd
import numpy as np

from datetime import datetime , timedelta
from typing import Callable
from .silence import Silence

class Duration:
    """Duration class, used to calculate the duration of the input or the start time"""
    def __init__(self , duration : int | float | timedelta | None = None , since : float | datetime | None = None):
        assert duration is not None or since is not None , "duration or since must be provided"
        assert duration is None or since is None , f"duration and since cannot be provided at the same time, got duration = {duration} and since = {since}"
        if duration is not None:
            if isinstance(duration , timedelta):
                dur = duration.total_seconds()
            else:
                dur = duration
        elif since is not None:
            if isinstance(since , datetime):
                dur = (datetime.now() - since).total_seconds()
            else:
                dur = (datetime.now() - datetime.fromtimestamp(since)).total_seconds()
        assert dur >= 0 , f"duration must be a positive duration , but got {dur}"
        self.duration = dur
    def __repr__(self):
        return self.fmtstr
    @property
    def hours(self):
        """Get the duration in hours"""
        return self.duration / 3600
    @property
    def minutes(self):
        """Get the duration in minutes"""
        return self.duration / 60
    @property
    def seconds(self):
        """Get the duration in seconds"""
        return self.duration
    @property
    def days(self):
        """Get the duration in days"""
        return self.duration / 86400
    @property
    def fmtstr(self):
        """Get the duration in a human-readable string"""
        # Calculate time components
        
        # Store components in a dictionary for f-string formatting
        if self.duration < 1:
            return '<1 Sec'
        elif self.duration < 60:
            return f'{self.duration:.1f} Sec'
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
    """simple timer to count time"""
    def __init__(self , *args , newline = False , exit_only = True , silent = False): 
        self.newline = newline
        self.exit_only = exit_only
        self.silent = silent
        self.key = '/'.join(args)
    def __enter__(self):
        self._init_time = datetime.now()
        if not self.silent and not Silence.silent and not self.exit_only: 
            print(self.enter_str , end='\n' if self.newline else '')
    def __exit__(self, type, value, trace):
        if not self.silent and not Silence.silent:
            print(self.exit_str)

    @property
    def enter_str(self):
        """Get the enter string"""
        return f'{self.key} start ... '
    @property
    def exit_str(self):
        """Get the exit string"""
        text = f'finished! Cost {Duration(since = self._init_time)}'
        if self.exit_only:
            return f'{self.key} {text}'
        elif self.newline:
            return text

class PTimer:
    """process timer , call to record and .summarize() to display the summary"""
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {} if record else None
    class ptimer:
        """process timer class, used to record the time of the input function"""
        def __init__(self , target : dict[str,list[float]] | None , key):
            self.target , self.key = target , key
            if self.target is not None and key not in self.target.keys(): 
                self.target[self.key] = []
        def __enter__(self):
            if self.target is not None: 
                self._init_time = datetime.now()
        def __exit__(self, type, value, trace):
            if self.target is not None: 
                self.target[self.key].append((datetime.now() - self._init_time).total_seconds())

    def func_timer(self , func : Callable):
        """timer wrapper for a function"""
        def wrapper(*args , **kwargs):
            with self.ptimer(self.recorder , func.__name__):
                return func(*args , **kwargs)
        return wrapper if self.recording else func

    def __call__(self , *args):
        return self.ptimer(self.recorder , '/'.join(args))
    
    def summarize(self):
        """summarize the recorded time"""
        if self.recorder is not None:
            tb = pd.DataFrame([[k , len(self.recorder[k]) , np.sum(self.recorder[k])] for k in self.recorder.keys()] ,
                                columns = pd.Index(['keys' , 'num_calls', 'total_time']))
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))
