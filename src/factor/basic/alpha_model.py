
import numpy as np
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Any , Literal

@dataclass
class Amodel:
    date  : int
    alpha : np.ndarray
    secid : np.ndarray
    name  : str = 'alpha0'

    def __post_init__(self):
        assert self.alpha.ndim == self.secid.ndim == 1 , (self.alpha.shape , self.secid.shape)
        assert self.alpha.shape == self.secid.shape , (self.alpha.shape , self.secid.shape)
    def __len__(self): return len(self.alpha)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},date={self.date},length={len(self)})'
    def copy(self): return deepcopy(self)
    def align(self , secid : np.ndarray | Any = None , inplace = False , nan = 0.):
        if secid is None: return self
        new_alpha = self if inplace else self.copy()
        value = np.full(len(secid) , nan , dtype=float)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        value[p0s] = new_alpha.alpha[p1s]
        new_alpha.alpha = value
        new_alpha.secid = secid
        return new_alpha

    @classmethod
    def create_random(cls , date : int , secid : np.ndarray | Any = None):
        assert secid is not None , 'When create random Amodel, secid must be submitted too!'
        return cls(date , np.random.randn(len(secid)) , secid , 'random_alpha')

    @classmethod
    def from_array(cls , date : int , data : np.ndarray , secid : np.ndarray | Any = None , name : str = 'given_alpha'):
        assert secid is not None , 'When submit alpha as np.ndarray, secid must be submitted too!'
        assert len(data) == len(data) , f'alpha must match secid, but get <{len(data)}> and <{len(data)}>'
        return cls(date , data , secid , name)

    @classmethod
    def from_dataframe(cls , date : int , data : pd.Series | pd.DataFrame , 
                       secid : np.ndarray | Any = None , name : str | Any = None):
        if isinstance(data , pd.Series): data = data.to_frame()
        if np.isin(['secid' , 'date'] , data.index.names).all():
            data = data.xs(date , level='date')
        else:
            if not isinstance(data.index , pd.RangeIndex): data = data.reset_index()
            if 'date' in data.columns: 
                data = data[data['date'] == date]
                assert len(data) , f'no data of date {date}!'
                data = data.drop(columns=['date'])
            data = data.set_index(['secid'])

        assert len(data.columns) == 1, f'When submit alpha as pd.DataFrame, there must be only one possible column :{data.columns}'
        if secid is not None: 
            data = data.reindex(secid).fillna(0)
        else:
            data = data.dropna()
        
        alpha = data.to_numpy().squeeze()
        secid = data.index.values
        if name is None: name = data.columns.values[0]
        return cls(date , alpha , secid , name)

    @classmethod
    def create(cls , date : int , data: np.ndarray | pd.DataFrame | pd.Series | Literal['random'] , secid : np.ndarray | Any = None):
        if isinstance(data , str) and data == 'random':
            return cls.create_random(date , secid)
        elif isinstance(data , np.ndarray):
            return cls.from_array(date , data , secid)
        else:
            return cls.from_dataframe(date , data , secid)
    
class AlphaModel:
    def __init__(self , name : str = 'Alpha0' , models : Amodel | list[Amodel] | dict[int,Amodel] | Any = None) -> None:
        self.name = name
        self.models : dict[int,Amodel] = {}
        self.append(models)

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.models)} days loaded)'

    @classmethod
    def from_dataframe(cls , data: pd.DataFrame | pd.Series , name = 'Alpha0'):
        if isinstance(data , pd.Series): data = data.to_frame()
        if not isinstance(data.index , pd.RangeIndex): data = data.reset_index()
        assert 'secid' in data and 'date' in data , data.columns
        models = [Amodel.from_dataframe(date , data) for date in data['date'].unique()]
        return cls(name , models)

    def append(self , amodel : Amodel | list[Amodel] | dict[int,Amodel] , override = True):
        if isinstance(amodel , Amodel):
            assert override or (amodel.date not in self.models.keys()) , amodel.date
            self.models[amodel.date] = amodel
        elif isinstance(amodel , list):
            for am in amodel: self.append(am , override=override)
        elif isinstance(amodel , dict):
            for am in amodel.values(): self.append(am , override=override)

    def available_dates(self): return np.array(list(self.models.keys()))

    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1
    
    def get(self , date : int , latest = True) -> Amodel | Any:
        use_date = self.latest_avail_date(date) if latest else date
        rmodel = self.models.get(use_date , None)

        return rmodel
    