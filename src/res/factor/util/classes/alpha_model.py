from __future__ import annotations
import numpy as np
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from scipy.stats import rankdata
from typing import Any , Literal , Callable , Union , Sequence

from src.proj import Const , DB
from src.data import DATAVENDOR
from src.func.transform import fill_na_as_const , winsorize_by_dist , zscore

from .general_model import GeneralModel

__all__ = ['AlphaModel' , 'Amodel' , 'AlphaComposite' , 'AlphaScreener']

def _rank_pct(arr : np.ndarray , axis : int = -1) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        ranks = rankdata(arr, method='average', axis=axis, nan_policy='omit')
        n_valid = np.count_nonzero(~np.isnan(arr), axis=axis, keepdims=True)
        rank_pct = np.where(np.isnan(arr) , np.nan , np.where(n_valid > 1, (ranks - 1) / (n_valid - 1), 0.5))
    return rank_pct

def _to_dates(date : int | list[int] | np.ndarray) -> np.ndarray:
    if not isinstance(date , (list , np.ndarray)):
        date = [date]
    return np.array(date)

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
    def zscore(self , inplace = False):
        new = self if inplace else self.copy()
        new.alpha = zscore(new.alpha)
        return new
    def pre_process(self , inplace = False):
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

    def to_alpha_model(self) -> AlphaModel:
        return AlphaModel(self.name , self)

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
    def combine_linear(
        cls , alphas : list[Amodel] , weights : list[float] | np.ndarray | None = None , 
        date : int | None = None , name : str = 'combined_alpha' , normalize = True
    ):
        assert any(not alpha.empty for alpha in alphas) , 'alphas must have at least one non-empty alpha'
        if len(alphas) == 1:
            return alphas[0].zscore(inplace=False) if normalize else alphas[0]

        if weights is None:
            weights = np.ones(len(alphas))
        elif isinstance(weights , list):
            weights = np.array(weights)
        assert len(alphas) == len(weights) , f'alphas and weights must have the same length, but got {len(alphas)} and {len(weights)}'

        if date is None:
            date = alphas[0].date

        secid = np.unique(np.concatenate([alpha.secid for alpha in alphas]))
        all_alphas = np.stack([alpha.alpha_of(secid) for alpha in alphas] , axis = 0)
        alpha = np.sum(all_alphas * weights[:,None] , axis = 0) / weights.sum()
        amodel = cls(date , alpha , secid , name)
        if normalize: 
            amodel = amodel.zscore(inplace=True)
        return amodel

    @classmethod
    def combine_worst(
        cls , alphas : list[Amodel] , method : Literal['worst' , 'worst2'] , 
        date : int | None = None , name : str = 'combined_alpha' , normalize = False
    ):
        assert any(not alpha.empty for alpha in alphas) , 'alphas must have at least one non-empty alpha'
        if len(alphas) == 1:
            return alphas[0].zscore(inplace=False) if normalize else alphas[0]

        if date is None:
            date = alphas[0].date

        secid = np.unique(np.concatenate([alpha.secid for alpha in alphas]))
        all_alphas = np.stack([alpha.alpha_of(secid) for alpha in alphas] , axis = 0)
        rank_pct = _rank_pct(all_alphas , axis = 1)
        
        if method == 'worst':
            has_data = np.any(~np.isnan(rank_pct), axis=0)
            alpha = _rank_pct(np.nan_to_num(rank_pct,nan=np.inf).min(axis = 0),axis = 0)
            alpha[~has_data] = np.nan
        elif method == 'worst2':
            if rank_pct.shape[0] <= 2:
                partitioned = rank_pct
            else:
                partitioned = np.partition(rank_pct, kth=2, axis=0)
            has_data = np.any(~np.isnan(partitioned[:2]), axis=0)
            alpha = np.full(partitioned.shape[1], np.nan)
            alpha[has_data] = np.nanmean(partitioned[:2][:,has_data], axis=0)
        else:
            raise ValueError(f'method must be "worst" or "worst2": {method}')
        amodel = cls(date , alpha , secid , name)
        if normalize: 
            amodel = amodel.zscore(inplace=True)
        return amodel

    @classmethod
    def empty_model(cls , date : int , name : str = 'empty_alpha') -> Amodel:
        return cls(date , np.array([]) , np.array([]) , name)

class AlphaModel(GeneralModel):
    '''Alpha model instance, contains alpha for multiple days'''
    def __init__(self , name : str = 'Alpha0' , models : Amodel | list[Amodel] | dict[int,Amodel] | Any = None) -> None:
        self.name = name
        self.models : dict[int,Amodel] = {}
        self.append(models)
    def rename(self , name : str):
        self.name = name
        for model in self.models.values():
            model.name = name
        return self
    def load_day_model(self, date: int) -> Any:
        """load alpha model for a specific date , is not implemented in this class"""
        ...
    def get_model(self , date : int , latest = True) -> Amodel:
        return self.get(date , latest)
    @property
    def empty(self) -> bool:
        return len(self.models) == 0 or (len(self.models) == 1 and self.item().empty)
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

    def append(self , model : Amodel | list[Amodel] | dict[int,Amodel] | AlphaModel | None , override = True):
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
    
    def subset(self , date : int | list[int] | np.ndarray , latest = False):
        if not isinstance(date , (list , np.ndarray)):
            date = [date]
        if latest:
            date = list(set([self.latest_avail_date(d) for d in date]))
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

ComponentInputType = Union[str , AlphaModel , Amodel]
class AlphaComponent:
    _cache_normalized : dict[str , AlphaModel] = {}
    _cache_unnormalized : dict[str , AlphaModel] = {}

    def __init__(self , input : ComponentInputType , normalize : bool = True):
        self.input = input
        self.normalize = normalize
        if isinstance(input , (AlphaModel , Amodel)):
            self.name = input.name
        else:
            if input in Const.Model.strategies['prediction']:
                alpha_type , alpha_name , alpha_column = 'pred' , input , None
            elif '@' in input:
                exprs = input.split('@')
                alpha_type , alpha_name = exprs[:2]
                alpha_column = exprs[2] if len(exprs) > 2 else None
            else:
                raise Exception(f'{input} is not a valid alpha')

            self.name = f'{alpha_type}@{alpha_name}'
            if alpha_type in ['sellside' , 'pred']:
                self.loads = self.db_loads(alpha_type , alpha_name , alpha_column , name = self.name , normalize = normalize)
            elif alpha_type == 'factor':
                self.loads = self.factor_loads(alpha_name , name = self.name , normalize = normalize)
            else:
                raise Exception(f'{alpha_type} is not a valid alpha type')

    def __repr__(self) -> str:
        return self.name

    def get_alpha_model(self , date : int | list[int] | np.ndarray) -> AlphaModel:
        if isinstance(self.input , AlphaModel):
            return self.input
        elif isinstance(self.input , Amodel):
            return self.input.to_alpha_model()

        if isinstance(date , int):
            date = [date]
        cached_alpha_model = self.get_cache(self.name , normalize = self.normalize)
        cached_dates = cached_alpha_model.available_dates() if cached_alpha_model else []
        new_dates = np.setdiff1d(date , cached_dates)
        
        if len(new_dates) == 0:
            return cached_alpha_model.subset(date , latest = True)
            
        alpha_model = self.loads(new_dates)
        new_alpha_model = cached_alpha_model.append(alpha_model)
        self.set_cache(new_alpha_model , self.name , normalize = self.normalize)
        return new_alpha_model.subset(date , latest = True)

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
    def db_loads(cls , db_src : str , db_key : str , db_column : str | None = None , name : str = 'alpha0' , normalize : bool = True) -> Callable[[int | list[int] | np.ndarray], AlphaModel]:
        from src.res.factor.util.classes import StockFactor
        def wrapper(date : int | list[int] | np.ndarray) -> AlphaModel:
            if not isinstance(date , (list , np.ndarray)):
                date = [date]
            column = db_column if db_column is not None else db_key
            df = DB.loads(db_src , db_key , date , override_existing_key = True , vb_level = 'never')
            if df.empty or min(date) < min(df['date']):
                prev_df = DB.load(db_src , db_key , min(date) , closest = True , vb_level = 'never').assign(date = min(date))
                df = prev_df if df.empty else pd.concat([prev_df , df])
            assert df.empty or (column in df.columns.to_list()) , f'{column} not in {df.columns} at date {date}'
            df = pd.DataFrame(columns=['secid' , 'date' , column]) if df.empty else df.loc[:,['secid' , 'date' , column]]
            factor = StockFactor(df)
            if normalize:
                factor = factor.normalize(inplace=True)
            alpha_model = factor.alpha_model()
            alpha_model.rename(name)
            return alpha_model
        return wrapper

    @classmethod
    def factor_loads(cls , factor_name : str , name : str = 'alpha0' , normalize : bool = True) -> Callable[[int | list[int] | np.ndarray], AlphaModel]:
        from src.res.factor.calculator import StockFactorHierarchy
        from src.res.factor.util.classes import StockFactor
        def wrapper(date : int | list[int] | np.ndarray) -> AlphaModel:
            if not isinstance(date , (list , np.ndarray)):
                date = [date]
            factor = StockFactorHierarchy.get_factor(factor_name).Loads(date)
            factor = StockFactor(factor)
            if normalize:
                factor = factor.normalize(inplace=True)
            alpha_model = factor.alpha_model()
            alpha_model.rename(name)
            return alpha_model
        return wrapper

class AlphaCombination:
    def __init__(self , alpha : str | ComponentInputType | Sequence[ComponentInputType] , components : list[str] | None = None):
        if isinstance(alpha , str):
            if ',' in alpha:
                self.name = 'combined_alpha'
                assert not components , f'components are not allowed when alpha name suggest a composite alpha , {alpha} {components}'
                self.components : list[ComponentInputType] = list(alpha.split(','))
            else:
                self.name = alpha
                self.components = list(components) if components else [alpha]
        else:
            assert not components , f'components are not allowed when alpha is Amodel / AlphaModel / list , {alpha} {components}'
            self.name = 'combined_alpha'
            self.components = [*alpha] if isinstance(alpha , Sequence) else [alpha]

    @property
    def alpha_components(self) -> list[AlphaComponent]:
        if not hasattr(self , '_alpha_components'):
            self._alpha_components = [AlphaComponent(component) for component in self.components]
        return self._alpha_components

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},components={self.components})'

    def __len__(self) -> int:
        return len(self.components)

    def get_alpha_models(self , date : int | list[int] | np.ndarray , 
                         other_models : str | ComponentInputType | Sequence[ComponentInputType] | None = None) -> list[AlphaModel]:
        date = _to_dates(date)
        alpha_models = [comp.get_alpha_model(date) for comp in self.alpha_components]

        if other_models:
            other_combination = AlphaCombination(other_models)
            alpha_models.extend([comp.get_alpha_model(date) for comp in other_combination.alpha_components])

        return alpha_models

class AlphaComposite(AlphaCombination):
    def __init__(self , alpha : ComponentInputType | Sequence[ComponentInputType] , components : list[str] | None = None , 
                 weights : list[float] | Literal['equal'] | None = None):
        super().__init__(alpha , components)
        
        if isinstance(weights , list):
            assert (len(weights) - len(self.components)) in [0 , 1], f'components {self.components} and weights {weights} must have the same length or the difference is 1'
            self.weights = np.array(weights).astype(float)
        elif weights is None or isinstance(weights , str) and weights == 'equal':
            self.weights = np.ones(len(self.components) + 1)
        else:
            raise ValueError(f'weights must be a list of floats or "equal": {weights}')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},components={self.components},weights={self.weights})'

    def get(self , date : int | list[int] | np.ndarray , other_models : str | ComponentInputType | Sequence[ComponentInputType] | None = None) -> AlphaModel:
        date = _to_dates(date)
        alpha_models = self.get_alpha_models(date , other_models)
        assert alpha_models , f'no alpha models created, if AlphaComposite is empty, please submit a final alpha or components'

        models : list[Amodel] = []
        for d in date:
            alphas = [alpha.get(d , latest=True) for alpha in alpha_models]
            if not any(not alpha.empty for alpha in alphas):
                for model in alpha_models:
                    print(model.name)
                    print(model.available_dates())
                raise ValueError(f'all alphas are empty at date {d}')
            amodel = Amodel.combine_linear(alphas , self.weights[:len(alpha_models)] , date = d)
            models.append(amodel)
        return AlphaModel(self.name , models)

class AlphaScreener(AlphaCombination):
    def __init__(self , alpha : ComponentInputType | Sequence[ComponentInputType] , components : list[str] | None = None ,
                 method : Literal['worst' , 'worst2'] = 'worst2' , ratio : float = 0.5):
        super().__init__(alpha , components)
        self.method : Literal['worst' , 'worst2'] = method
        self.ratio = ratio

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},components={self.components},method={self.method})'

    def get(self , date : int | list[int] | np.ndarray , other_models : str | ComponentInputType | Sequence[ComponentInputType] | None = None) -> AlphaModel | None:
        date = _to_dates(date)
        alpha_models = self.get_alpha_models(date , other_models)
        if not alpha_models:
            return None

        models : list[Amodel] = []
        for d in date:
            amodel = Amodel.combine_worst([alpha.get(d , latest=True) for alpha in alpha_models] , self.method , date = d)
            models.append(amodel)
        return AlphaModel(self.name , models)

    def screened_pool(self , date : int , secid : np.ndarray | list[int] | Any = None , ratio : float | None = None , 
                      other_models : str | ComponentInputType | Sequence[ComponentInputType] | None = None) -> np.ndarray | None:
        secid = np.array(secid) if secid is not None else None
        alpha = self.get(date , other_models)
        if alpha is None or alpha.empty:
            return secid

        alpha = alpha.get(date).to_dataframe()
        if alpha.empty:
            return secid
        
        if secid is not None and len(secid) > 0: 
            alpha = alpha.query('secid in @secid').copy()
        alpha.loc[:, 'rankpct'] = alpha['alpha'].rank(pct = True , method = 'first' , ascending = True).fillna(0)
        ratio = ratio if ratio is not None else self.ratio
        return alpha.query('rankpct >= @ratio')['secid'].to_numpy()