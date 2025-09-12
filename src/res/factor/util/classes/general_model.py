import numpy as np

from abc import ABC , abstractmethod
from copy import deepcopy
from typing import Any

__all__ = ['GeneralModel']

class GeneralModel(ABC):
    '''Any model composed of daily models. Must define abstractmethod: __init__ , load_day_model'''
    @abstractmethod
    def __init__(self , *args , **kwargs) -> None:
        self.models : dict[int,Any] = {}
        self.name : str = 'GeneralModel'
    @abstractmethod
    def load_day_model(self , date : int) -> Any: ...
    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.models)} days loaded)'
    def __len__(self): 
        return len(self.models)
    def __bool__(self): 
        return bool(self.models)
    def append(self , model : Any , override = False):
        assert override or (model.date not in self.models.keys()) , model.date
        self.models[model.date] = model
        return self
    def available_dates(self): 
        return np.array(list(self.models.keys()))
    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: 
            return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1
    def get_model(self , date : int , latest = True):
        return self.get(date , latest)
    def get(self , date : int , latest = True):
        use_date = self.latest_avail_date(date) if latest else date
        if use_date not in self.models and use_date in self.available_dates():
            self.append(self.load_day_model(date))
        return self.models.get(use_date , None)
    def has(self , date : int , latest = True):
        if date in self.models: 
            return True
        return self.available_dates().min() <= date if latest else False
    def load_models(self , dates : np.ndarray | Any = None , start : int = -1 , end : int = -1):
        if dates is None:
            dates = self.available_dates()
        dates = dates[(dates >= start) & (dates <= end)]
        [self.append(self.load_day_model(date)) for date in dates if date not in self.models]
    def copy(self): return deepcopy(self)
    def item(self):
        assert len(self.models) == 1 , f'expect 1 model , but got {len(self.models)}'
        return list(self.models.values())[0]
