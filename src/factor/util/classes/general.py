import numpy as np
import pandas as pd

from abc import ABC , abstractmethod
from copy import deepcopy
from typing import Any , Literal , Optional

from src.basic.conf import EPS_WEIGHT
from src.data import DATAVENDOR

__all__ = ['GeneralModel' , 'Port']

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
    def __len__(self): return len(self.models)
    def __bool__(self): return bool(self.models)
    def append(self , model : Any , override = False):
        assert override or (model.date not in self.models.keys()) , model.date
        self.models[model.date] = model
        return self
    def available_dates(self): return np.array(list(self.models.keys()))
    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: return date
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
        if date in self.models: return True
        return self.available_dates().min() <= date if latest else False
    def load_models(self , dates : np.ndarray | Any = None , start : int = -1 , end : int = -1):
        if dates is None:
            dates = self.available_dates()
        dates = dates[(dates >= start) & (dates <= end)]
        [self.append(self.load_day_model(date)) for date in dates if date not in self.models]
    def copy(self): return deepcopy(self)

class Port:
    '''portfolio realization of one day'''

    def __init__(self , port : Optional[pd.DataFrame] , date : int | Any = -1 , 
                 name : str | Any = 'default' , value : float = 1.) -> None:
        self.exists = port is not None 
        if port is None or port.empty:
            port = pd.DataFrame(columns=['secid','weight']).astype({'secid':int,'weight':float})
        else:
            port = port.groupby('secid')['weight'].sum().reset_index()
        self.port = port[port['weight'] != 0]
        self.date = date
        self.name = name
        self.value = value
        self.sort()

    def __bool__(self): return self.exists
    def __repr__(self): 
        return '\n'.join([f'Portfolio <date={self.date}> <name={self.name}> <value={self.value}>: ', str(self.port)])
    def __add__(self , other): return self.merge(other)
    def __mul__(self , other):  return self.rescale(other)
    def __rmul__(self , other):  return self.rescale(other)
    def __truediv__(self , other): return self.rescale(1 / other)
    def __sub__(self , other): return self.merge(other * -1.)

    def sort(self , by : Literal['secid' , 'weight'] = 'weight' , ascending=False):
        self.port = self.port.sort_values(by , ascending=ascending).reset_index(drop=True)
        return self

    def is_emtpy(self): return not self.exists or self.port.empty

    def forward(self , n : int = 1 , inplace = True):
        if n == 0: return self if inplace else self.copy()
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert n > 0 , f'n must be non-negative! ({n})'
        return self.evolve_to_date(DATAVENDOR.td(self.date , n).td , inplace)

    def backward(self , n : int = -1 , inplace = True):
        if n == 0: return self if inplace else self.copy()
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert n < 0 , f'n must be non-positive! ({n})'
        return self.evolve_to_date(DATAVENDOR.td(self.date , n).td , inplace)
    
    def evolve_to_date(self , date : int | Any , inplace = False , rebalance = False):
        rslt = self if inplace else self.copy()
        if date == rslt.date: return rslt

        old_date = DATAVENDOR.td(self.date).td
        new_date = DATAVENDOR.td(date).td
        if old_date == new_date: 
            rslt.date = date
            return rslt

        old_pos   = rslt.long_position , rslt.short_position
        old_value = rslt.value

        q = DATAVENDOR.get_quote_ret(old_date , new_date)
        assert q is not None, f'Ret Quote (at {new_date}) is does not exists'
        port = rslt.port.merge(q , on = 'secid')
        port['new_weight'] = port['weight'] * (1 + port['ret'])

        rslt.date = date
        rslt.value = old_value * (1. + port['new_weight'].sum() - port['weight'].sum())
        port['weight'] = port['new_weight'] * old_value / rslt.value
        rslt.port = port.loc[:,['secid' , 'weight']].sort_values('weight' , ascending=False)
        if rebalance: rslt.rebalance(*old_pos)
        return rslt
    
    def fut_ret(self , new_date : Optional[int] = None) -> float:
        if not self: return 0.
        old_date = DATAVENDOR.td(self.date).td
        if new_date is None: new_date = DATAVENDOR.td(old_date , 1).td

        q = DATAVENDOR.get_quote_ret(old_date , new_date)
        assert q is not None, f'Ret Quote (at {new_date}) is does not exists'
        port = self.port.merge(q , on = 'secid').fillna(0)
        return (port['weight'] * port['ret']).to_numpy().sum()
    
    def rebalance(self , long_position : float = 1., short_position : float = 0.):
        assert long_position >= 0 and short_position >= 0 , (long_position , short_position)
        L = self.port['weight'] >= 0
        S = ~L
        if L.any(): self.port.loc[L , 'weight'] *=   long_position / self.port.loc[L,'weight'].sum()
        if S.any(): self.port.loc[S , 'weight'] *= -short_position / self.port.loc[S,'weight'].sum()
        return self

    @classmethod
    def create(cls , secid : np.ndarray | Any , weight : np.ndarray | Any , **kwargs):
        weight = weight * ((weight >= EPS_WEIGHT) + (weight <= -EPS_WEIGHT))
        df = pd.DataFrame({'secid':secid , 'weight' : weight})
        df = df[df['weight'] != 0]
        return cls(df , **kwargs)
    
    @classmethod
    def rand_port(cls , date : int | None = None , name : str = 'rand_port' , size : int = 30):
        date = np.random.randint(20200101 , 20231231) if date is None else date
        port = pd.DataFrame({'secid' : np.random.choice(DATAVENDOR.secid(date) , size) , 'weight' : np.random.rand(size)})
        return cls(port , date , name).rescale()

    @classmethod
    def none_port(cls , date : int , name = 'none'): return cls(None , date , name)
    
    @property
    def port_with_date(self):
        if len(self.port):
            return pd.DataFrame({'date':self.date , **self.port})
        else:
            return pd.DataFrame(columns=['date','secid','weight']).astype({'date':int,'secid':int,'weight':float})
        
    @property
    def full_table(self):
        if len(self.port):
            return self.port.assign(name = self.name , date = self.date)[['name' , 'date' , 'secid' , 'weight']]
        else:
            return pd.DataFrame()
        
    @property
    def secid(self): return self.port['secid'].to_numpy()
    @property
    def weight(self): return self.port['weight'].to_numpy()
    @property
    def position(self): return self.port['weight'].sum()
    @property
    def holdings(self): return self.value * self.weight
    @property
    def long_position(self): return self.port[self.port['weight'] > 0]['weight'].sum()
    @property
    def short_position(self): return -self.port[self.port['weight'] < 0]['weight'].sum()
    def copy(self): return deepcopy(self)
    def rename(self , new_name : str):
        self.name = new_name
        return self
    def rescale(self , scale = 1. , inplace = False): 
        new = self if inplace else self.copy()
        new.port['weight'] = scale * new.port['weight'] / new.position
        return new
    def weight_align(self , secid , fillna = 0.):
        return self.port.set_index('secid').reindex(secid)['weight'].fillna(fillna).to_numpy()
    def merge(self , another , name = None , inplace = False):
        assert isinstance(another , Port) , another
        new = self if inplace else self.copy()
        if name is not None: new.name = name
        if not new.is_emtpy() and not another.is_emtpy():
            combined = pd.concat([new.port, another.port], ignore_index=True).groupby('secid', as_index=False)['weight'].sum()
        elif new.is_emtpy():
            combined = another.port.copy()
        else:
            combined = new.port
        new.port = combined
        return new
    def turnover(self , another):
        if not self or self is another: return 0.
        assert isinstance(another , Port) , another
        return (self - another).port['weight'].abs().sum()