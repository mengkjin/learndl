
import numpy as np
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Any , Literal , Optional

from ..classes import GeneralModel
from ...func.transform import fill_na_as_const , winsorize_by_dist , zscore

@dataclass
class Amodel:
    '''Alpha model of one day instance'''
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
    def assign(self , date : Optional[int] = None , name : Optional[str] = None):
        if date is not None: self.date = date
        if name is not None: self.name = name
        return self
    def preprocess(self , inplace = False):
        # nan_as_num , winsor , normal
        new = self if inplace else self.copy()
        new.alpha = zscore(winsorize_by_dist(fill_na_as_const(new.alpha) , winsor_rng=0.5))
        return new

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
    
class AlphaModel(GeneralModel):
    '''Alpha model instance, contains alpha for multiple days'''
    def __init__(self , name : str = 'Alpha0' , models : Amodel | list[Amodel] | dict[int,Amodel] | Any = None) -> None:
        self.name = name
        self.models : dict[int,Amodel] = {}
        self.append(models)
    def load_day_model(self, date: int) -> Any:
        # do something here
        ...
    def __repr__(self):
        return f'{self.__class__.__name__} (name={self.name})({len(self.models)} days loaded)'
    @classmethod
    def from_dataframe(cls , data: pd.DataFrame | pd.Series , name : str | Any = None):
        if isinstance(data , pd.Series): data = data.to_frame()
        if not isinstance(data.index , pd.RangeIndex): data = data.reset_index()
        assert 'secid' in data and 'date' in data , data.columns
        models = [Amodel.from_dataframe(date , data) for date in data['date'].unique()]
        assert models , f'no models created'
        return cls(name if name else models[0].name , models)

    def append(self , model : Amodel | list[Amodel] | dict[int,Amodel] , override = True):
        if isinstance(model , Amodel):
            assert override or (model.date not in self.models.keys()) , model.date
            self.models[model.date] = model
        elif isinstance(model , list):
            for am in model: self.append(am , override=override)
        elif isinstance(model , dict):
            for am in model.values(): self.append(am , override=override)

    def get(self , date : int , latest = True , lag : int = 0) -> Amodel | Any:
        if lag:
            assert lag > 0 , lag
            avail_dates = np.sort(self.available_dates())
            avail_dates = avail_dates[avail_dates < date]
            if len(avail_dates): date = avail_dates[-min(lag , len(avail_dates))]
        model = super().get(date , latest)
        assert model is None or isinstance(model , Amodel)
        return model
    
    def lag_all_models(self , lag_period : int = 0 , inplace = False , rename = True):
        new = self if inplace else self.copy()
        if rename: new.name = f'{new.name}.lag{lag_period}'
        if lag_period == 0: return new
        dates = np.sort(new.available_dates())[::-1]
        for i , date in enumerate(dates):
            tar_date = dates[min(i+lag_period,len(dates)-1)]
            new.models[date] = new.models[tar_date].assign(date = date , name = new.name)
        return new
    