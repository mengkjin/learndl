
import numpy as np
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Any , Literal , Callable

from src.proj import Proj
from src.data import DATAVENDOR
from src.func.transform import fill_na_as_const , winsorize_by_dist , zscore

from .general_model import GeneralModel

__all__ = ['AlphaModel' , 'Amodel']

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
    def __len__(self): 
        return len(self.alpha)
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},date={self.date},length={len(self)})'
    def __bool__(self):
        return True
    @property
    def empty(self) -> bool:
        return len(self) == 0
    def get_model(self , *args , **kwargs): 
        return self
    def copy(self): 
        return deepcopy(self)
    def align(self , secid : np.ndarray | Any = None , inplace = False , nan = 0.):
        new_alpha = self if inplace else self.copy()
        if secid is None: 
            return self
        if self.empty:
            self.secid = secid
            self.alpha = np.full(len(secid) , nan , dtype=float)
            return self
       
        value = np.full(len(secid) , nan , dtype=float)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        value[p0s] = new_alpha.alpha[p1s]
        new_alpha.alpha = value
        new_alpha.secid = secid
        return new_alpha
    def assign(self , date : int | None = None , name : str | None = None):
        if date is not None: 
            self.date = date
        if name is not None: 
            self.name = name
        return self
    def preprocess(self , inplace = False):
        # nan_as_num , winsor , normal
        new = self if inplace else self.copy()
        new.alpha = zscore(winsorize_by_dist(fill_na_as_const(new.alpha) , winsor_rng=0.5))
        return new
    def alpha_of(self , secid : np.ndarray | Any = None , nan = 0. , rank = False) -> np.ndarray:
        if self.empty:
            return self.alpha if secid is None else np.full(len(secid) , nan , dtype=float)
        value = self.alpha if not rank else pd.Series(self.alpha).rank(pct=True).to_numpy()
        if secid is None: 
            return value
        new_value = np.full(len(secid) , nan , dtype=float)
        _ , p0s , p1s = np.intersect1d(secid , self.secid , return_indices=True)
        new_value[p0s] = value[p1s]
        return new_value

    def to_dataframe(self , indus = False , na_indus_as : Any = None):
        df = pd.DataFrame({'secid' : self.secid , 'alpha' : self.alpha})
        if indus: 
            df = DATAVENDOR.INFO.add_indus(df , self.date , na_indus_as)
        return df

    @classmethod
    def create_random(cls , date : int , secid : np.ndarray | list[int] | Any = [1,2,600001]):
        assert secid is not None , 'When create random Amodel, secid must be submitted too!'
        if isinstance(secid , list): 
            secid = np.array(secid)
        return cls(date , np.random.randn(len(secid)) , secid , 'random_alpha')

    @classmethod
    def from_array(cls , date : int , data : np.ndarray , secid : np.ndarray , name : str = 'given_alpha'):
        assert len(data) == len(data) , f'alpha must match secid, but get <{len(data)}> and <{len(data)}>'
        return cls(date , data , secid , name)

    @classmethod
    def from_dataframe(cls , date : int | Any , data : pd.DataFrame , 
                       secid : np.ndarray | Any = None , name : str | Any = None , filter_date = False):
        """data must include ['date' , 'secid' , name_of_alpha] 3 columns"""
        value_col = np.setdiff1d(data.columns.values , ['date' , 'secid'])
        assert len(value_col) == 1 , f'When submit alpha as pd.DataFrame, there must be only one possible column :{data.columns} but got {value_col}'
        if filter_date:
            data = data.query('date == @date')
        data = data.reset_index().set_index('secid')[value_col].dropna()
        
        if secid is not None: 
            data = data.reindex(secid).fillna(0)

        return cls(date , data.to_numpy().squeeze() , data.index.values , name or value_col[0])

    @classmethod
    def create(cls , date : int , data: np.ndarray | pd.DataFrame | pd.Series | Literal['random'] , secid : np.ndarray | Any = None):
        if isinstance(data , str) and data == 'random':
            return cls.create_random(date , secid)
        elif isinstance(data , np.ndarray):
            assert secid is not None , 'When submit alpha as np.ndarray, secid must be submitted too!'
            return cls.from_array(date , data , secid)
        else:
            if isinstance(data , pd.Series):
                data = data.to_frame()
            return cls.from_dataframe(date , data , secid , filter_date=True)
        
    @classmethod
    def combine(cls , alphas : list['Amodel'] , weights : list[float] | np.ndarray | None = None , 
                date : int | None = None , name : str = 'combined_alpha' , normalize = True):
        assert any(not alpha.empty for alpha in alphas) , 'alphas must have at least one non-empty alpha'
        if weights is None:
            weights = np.ones(len(alphas))
        elif isinstance(weights , list):
            weights = np.array(weights)
        assert len(alphas) == len(weights) , f'alphas and weights must have the same length, but got {len(alphas)} and {len(weights)}'
        if len(alphas) == 1:
            return alphas[0]

        if date is None:
            date = alphas[0].date

        secid = np.unique(np.concatenate([alpha.secid for alpha in alphas]))
        all_alphas = np.stack([alpha.alpha_of(secid) for alpha in alphas] , axis = 0)
        alpha = np.sum(all_alphas * weights[:,None] , axis = 0) / weights.sum()
        if normalize: 
            alpha = zscore(alpha)
        return cls(date , alpha , secid , name)

    @classmethod
    def empty_model(cls , date : int , name : str = 'empty_alpha') -> 'Amodel':
        return cls(date , np.array([]) , np.array([]) , name)

class AlphaModel(GeneralModel):
    '''Alpha model instance, contains alpha for multiple days'''
    def __init__(self , name : str = 'Alpha0' , models : Amodel | list[Amodel] | dict[int,Amodel] | Any = None) -> None:
        self.name = name
        self.models : dict[int,Amodel] = {}
        self.append(models)
    def load_day_model(self, date: int) -> Any:
        """load alpha model for a specific date , is not implemented in this class"""
        ...
    def item(self) -> Amodel:
        return super().item()
    def alpha(self) -> np.ndarray:
        return self.item().alpha
    def __repr__(self):
        return f'{self.__class__.__name__} (name={self.name})({len(self.models)} days loaded)'
    @classmethod
    def from_dataframe(cls , data: pd.DataFrame | pd.Series , name : str | Any = None):
        if isinstance(data , pd.Series): 
            data = data.to_frame()
        if not isinstance(data.index , pd.RangeIndex): 
            data = data.reset_index()
        assert 'secid' in data and 'date' in data , data.columns
        models = [Amodel.from_dataframe(date , sub_data) for date , sub_data in data.groupby('date')]
        assert models or name , f'no models created, name must be submitted too!'
        if name is None:
            name = models[0].name
        return cls(name , models)

    def append(self , model : 'Amodel | list[Amodel] | dict[int,Amodel] | AlphaModel | None' , override = True):
        if model is None:
            ...
        elif isinstance(model , Amodel):
            assert override or (model.date not in self.models.keys()) , model.date
            self.models[model.date] = model
        elif isinstance(model , list):
            for am in model: 
                self.append(am , override=override)
        elif isinstance(model , dict):
            for am in model.values():
                self.append(am , override=override)
        elif isinstance(model , AlphaModel):
            assert model.name == self.name , f'model name must be the same as self.name: {model.name} != {self.name}'
            for am in model.models.values():
                self.append(am , override=override)
        else:
            raise ValueError(f'model must be a Amodel, list of Amodel, dict of Amodel, or AlphaModel: {type(model)}')
        return self
    
    def subset(self , date : int | list[int] | np.ndarray):
        if not isinstance(date , (list , np.ndarray)):
            date = [date]
        models = {d : self.models.get(d) for d in date if d in self.models}
        return self.__class__(name = self.name , models = models)

    def get(self , date : int , latest = True , lag : int = 0) -> Amodel:
        if lag:
            assert lag > 0 , lag
            avail_dates = np.sort(self.available_dates())
            avail_dates = avail_dates[avail_dates < date]
            if len(avail_dates): 
                date = avail_dates[-min(lag , len(avail_dates))]
        model = super().get(date , latest)
        if model is None:
            model = Amodel.empty_model(date , name = self.name)
        return model
    
    def lag_all_models(self , lag_period : int = 0 , inplace = False , rename = True):
        new = self if inplace else self.copy()
        if rename: 
            new.name = f'{new.name}.lag{lag_period}'
        if lag_period == 0: 
            return new
        dates = np.sort(new.available_dates())[::-1]
        for i , date in enumerate(dates):
            tar_date = dates[min(i+lag_period,len(dates)-1)]
            new.models[date] = new.models[tar_date].assign(date = date , name = new.name)
        return new

class CompositeAlpha:
    def __init__(self , name : str | list[str] , components : list[str] | None = None , weights : list[float] | Literal['equal'] | None = None):
        assert name , f'name must be non-empty: {name}'
        
        if isinstance(name , list) or ',' in name:
            assert not components , f'components are not allowed when alpha name suggest a composite alpha , {name} {components}'
            self.name = 'combined_alpha'
            self.components = name if isinstance(name , list) else name.split(',')
        else:
            self.name = name
            self.components = components if components else [name]
        
        if isinstance(weights , list):
            assert len(self.components) == len(weights) , f'components {self.components} and weights {weights} must have the same length'
            self.weights = np.array(weights).astype(float)
        elif weights is None or isinstance(weights , str) and weights == 'equal':
            self.weights = np.ones(len(self.components))
        else:
            raise ValueError(f'weights must be a list of floats or "equal": {weights}')

    @property
    def composite_components(self) -> list['CompositeAlphaComponent']:
        return [CompositeAlphaComponent(component) for component in self.components]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},components={self.components},weights={self.weights})'

    def get(self , date : int | list[int] | np.ndarray) -> AlphaModel:
        if not isinstance(date , (list , np.ndarray)):
            date = [date]

        alpha_models = [comp.get_alpha_model(date) for comp in self.composite_components]
        models : list[Amodel] = []
        for d in date:
            amodel = Amodel.combine([alpha.get(d , latest=True) for alpha in alpha_models] , self.weights , date = d)
            models.append(amodel)
        return AlphaModel(self.name , models)

class CompositeAlphaComponent:
    _cache_normalized : dict[str , AlphaModel] = {}
    _cache_unnormalized : dict[str , AlphaModel] = {}

    def __init__(self , name : str):
        self.name = name

        if name in Proj.Conf.Model.SETTINGS['prediction']:
            self.loads = self.db_loads('pred' , name)
        elif '@' in name:
            exprs = name.split('@')
            alpha_type , alpha_name = exprs[:2]
            alpha_column = exprs[2] if len(exprs) > 2 else alpha_name
            if alpha_type in ['sellside' , 'pred']:
                self.loads = self.db_loads(alpha_type , alpha_name , alpha_column)
            elif alpha_type == 'factor':
                self.loads = self.factor_loads(alpha_name)
            else:
                raise Exception(f'{alpha_type} is not a valid alpha type')
        else:
            raise Exception(f'{name} is not a valid alpha')

    def __repr__(self) -> str:
        return self.name

    def get_alpha_model(self , date : int | list[int] | np.ndarray , normalize : bool = True) -> AlphaModel:
        from src.res.factor.util.classes import StockFactor

        cached_alpha_model = self.get_cache(self.name , normalize = normalize)
        cached_dates = cached_alpha_model.available_dates() if cached_alpha_model else []
        new_dates = np.setdiff1d(date , cached_dates)
        
        factor = StockFactor(self.loads(new_dates))
        if normalize:
            factor = factor.normalize(inplace=True)
        alpha_model = factor.alpha_model()
        new_alpha_model = cached_alpha_model.append(alpha_model)
        self.set_cache(new_alpha_model , self.name , normalize = normalize)
        return new_alpha_model.subset(date)

    @classmethod
    def get_cache(cls , name : str , normalize : bool = True) -> AlphaModel:
        if normalize:
            alpha_model = cls._cache_normalized.get(name)
        else:
            alpha_model = cls._cache_unnormalized.get(name)
        if alpha_model is None:
            alpha_model = AlphaModel(name = name)
        return alpha_model

    @classmethod
    def set_cache(cls , alpha_model : AlphaModel , name : str , normalize : bool = True):
        if normalize:
            cls._cache_normalized[name] = alpha_model
        else:
            cls._cache_unnormalized[name] = alpha_model

    @classmethod
    def db_loads(cls , db_src : str , db_key : str , db_column : str | None = None) -> Callable[[int | list[int] | np.ndarray], pd.DataFrame]:
        from src.proj import DB
        def wrapper(date : int | list[int] | np.ndarray) -> pd.DataFrame:
            if not isinstance(date , (list , np.ndarray)):
                date = [date]
            column = db_column if db_column is not None else db_key
            df = DB.loads(db_src , db_key , date , vb_level = 'inf')
            if df.empty or min(date) < min(df['date']):
                df = pd.concat([DB.load(db_src , db_key , min(date) , closest = True , vb_level = 'inf').assign(date = min(date)) , df])
            assert df.empty or (column in df.columns.to_list()) , f'{column} not in {df.columns} at date {date}'
            df = pd.DataFrame(columns=['secid' , 'date' , column]) if df.empty else df.loc[:,['secid' , 'date' , column]]
            return df
        return wrapper

    @classmethod
    def factor_loads(cls , factor_name : str) -> Callable[[int | list[int] | np.ndarray], pd.DataFrame]:
        from src.res.factor.calculator import StockFactorHierarchy
        def wrapper(date : int | list[int] | np.ndarray) -> pd.DataFrame:
            if not isinstance(date , (list , np.ndarray)):
                date = [date]
            df = StockFactorHierarchy.get_factor(factor_name).Loads(date)
            if df.empty or min(date) < min(df['date']):
                df = pd.concat([StockFactorHierarchy.get_factor(factor_name).Load(min(date) , closest = True).assign(date = min(date)) , df])
            return df
        return wrapper