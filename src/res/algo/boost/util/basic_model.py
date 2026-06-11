"""Abstract base class and utilities shared by all boost model implementations.

Classes:
    BasicBoostModel  — ABC with data import, weight wiring, fit/predict interface,
                       and serialisation helpers.

Functions:
    load_xingye_data — Load the Xingye (兴业) factor dataset for offline testing.
"""
from __future__ import annotations
import torch
import numpy as np
import pandas as pd

from abc import ABC , abstractmethod
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any

from src.proj import PATH , DB , Load , Base
from src.func.metric import ic_2d , rankic_2d

from .dataset import BoostInput , BoostOutput
from .weight import BoostWeightMethod

__all__ = ['BasicBoostModel' , 'load_xingye_data']

class BasicBoostModel(ABC, Base.BoundLogger):
    """Abstract base class for all gradient-boost wrappers.

    Sub-classes must implement :meth:`fit`, :meth:`predict`, :meth:`to_dict`,
    and :meth:`load_dict`.

    Class attributes:
        DEFAULT_TRAIN_PARAM:         Default hyper-parameters for the underlying
                                     booster.  Sub-class overrides are merged at
                                     ``__init__`` time via :meth:`update_param`.
        DEFAULT_WEIGHT_PARAM:        Default :class:`BoostWeightMethod` kwargs.
        DEFAULT_CATEGORICAL_N_BINS:  Fallback bin count for softmax objectives
                                     when ``n_bins`` is not specified (default 3).
        DEFAULT_CATEGORICAL_MAX_BINS: Hard cap on ``n_bins`` for softmax
                                     (default 10).
    """
    DEFAULT_TRAIN_PARAM = {}
    DEFAULT_WEIGHT_PARAM = {}
    MASK_PARAM : tuple[str, ...] = ('rank_target_size',)
    DEFAULT_CATEGORICAL_N_BINS = 3
    DEFAULT_CATEGORICAL_MAX_BINS = 10

    def __repr__(self) -> str: 
        return f'{self.__class__.__name__}(train_param={self.train_param})'

    @cached_property
    def weight_keys(self) -> frozenset[str]:
        return frozenset(BoostWeightMethod.__slots__)
    @cached_property
    def train_param_keys(self) -> frozenset[str]:
        return frozenset(self.DEFAULT_TRAIN_PARAM.keys())

    def set_params(
        self , params : dict[str,Any] | None = None , cuda = True , seed = None , * , overrides : dict[str,Any] | None = None , **kwargs
    ):
        params = params or {}
        overrides = overrides or {}
        self.train_param : dict[str,Any] = deepcopy(self.DEFAULT_TRAIN_PARAM) | {k:v for k,v in params.items() if k in self.train_param_keys}
        self.train_param.update({k:v for k,v in overrides.items() if k in self.train_param_keys and v is not None})
        self.weight_param : dict[str,Any] = deepcopy(self.DEFAULT_WEIGHT_PARAM) | {k:v for k,v in params.items() if k in self.weight_keys}

        self.cuda = cuda
        self.seed = seed
        self.assert_param(**kwargs)
        return self
    
    def assert_param(self , **kwargs): 
        assert self.train_param['objective'] in ['mse', 'mae', 'rank'] , self.train_param['objective']
        assert all(k in self.DEFAULT_TRAIN_PARAM for k in self.train_param) , \
            f'{str([k for k in self.train_param if k not in self.DEFAULT_TRAIN_PARAM])} not in DEFAULT_TRAIN_PARAM'

    def get_param(self , key : str , default : Any = None):
        return self.train_param.get(key , self.DEFAULT_TRAIN_PARAM.get(key , default))
    
    def import_data(self , train : Any = None , valid : Any = None , test : Any = None):
        if train is not None: 
            self.data['train'] = self.to_boost_input(train , self.weight_param)
        if valid is not None: 
            self.data['valid'] = self.to_boost_input(valid , self.weight_param)
        if test  is not None: 
            self.data['test']  = self.to_boost_input(test , self.weight_param)
        return self

    def update_feature(self , use_feature = None):
        if use_feature is not None: 
            [bdata.set_data_param(use_feature = use_feature) for bdata in self.data.values()]
        return self

    @abstractmethod
    def fit(self , train : BoostInput | Any = None , valid : BoostInput | Any = None , silent = False , **kwargs):
        self.boost_fit_inputs(train , valid , silent)
        return self

    @cached_property
    def data(self) -> dict[str,BoostInput]:
        return {}

    @property
    def is_rankor(self) -> bool:
        return 'rank' in self.train_param['objective'].lower()
    
    @property
    def boost_objective_multi(self):
        obj = self.train_param.get('objective' , None)
        return obj is not None and 'softmax' in obj
    
    @property
    def boost_objective_rank(self):
        obj = self.train_param.get('objective' , None)
        return obj is not None and 'rank' in obj

    def boost_input(self , x : BoostInput | str | Any = 'test'):
        return self.data[x] if isinstance(x , str) else x

    def boost_fit_inputs(self , train : BoostInput | Any = None , valid : BoostInput | Any = None , silent = False , **kwargs):
        self.silent = silent

        train_param = {
            k:v for k,v in self.train_param.items() 
            if k in self.DEFAULT_TRAIN_PARAM and k not in self.MASK_PARAM and v is not None}

        # categorical_label
        n_bins : int | None = train_param.pop('n_bins', None)
        if self.boost_objective_multi:
            if n_bins is None: 
                n_bins = self.DEFAULT_CATEGORICAL_N_BINS
                if not self.silent: 
                    self.logger.alert1(f'n_bins not specified, using default value {self.DEFAULT_CATEGORICAL_N_BINS}')
            n_bins = min(n_bins , self.DEFAULT_CATEGORICAL_MAX_BINS)            
        train_param['n_bins'] = n_bins

        if train is None: 
            train = self.data['train']
        if valid is None: 
            valid = self.data['valid']

        self.fit_train_ds = train.set_data_param(n_bins = n_bins).Dataset()
        self.fit_valid_ds = valid.set_data_param(n_bins = n_bins).Dataset()
        self.fit_train_param = train_param

        return self
    
    @abstractmethod
    def predict(self , x : BoostInput | str | Any = 'test') -> BoostOutput:
        ...

    @abstractmethod
    def to_dict(self) -> dict[str,Any]: 
        model_dict = {
            'class_name' : self.__class__.__name__ ,
            'train_param' : self.train_param ,
            'weight_param' : self.weight_param
        }
        return model_dict

    @abstractmethod
    def load_dict(self , model_dict : dict[str,Any] , cuda = False , seed = None): 
        assert self.__class__.__name__ == model_dict['class_name'] , (self.__class__.__name__ , model_dict['class_name'])

        self.cuda = cuda
        self.seed = seed
        self.train_param = model_dict['train_param']
        self.weight_param = model_dict['weight_param']

        return self
    
    @classmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False):
        return cls().load_dict(model_dict , cuda)
    
    def test_result(self , test : BoostInput | Any = None , plot_path : Path | None = None):
        import matplotlib.pyplot as plt
        df = self.calc_ic(test)
        plt.figure()
        
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(df['ic'].mean() , df['rankic'].mean()))
        if plot_path:
            plot_path.mkdir(exist_ok=True)
            plt.savefig(plot_path.joinpath('test_prediction.png'),dpi=1200)
        return df
    
    def calc_ic(self , test : BoostInput | Any = None):
        if test is None: 
            test = self.data['test']
        label = test.y
        pred  = self.predict(test).pred
        index = test.date

        ic = ic_2d(pred , label , dim = 0) if label is not None else torch.full_like(pred[0,:] , fill_value=torch.nan)
        ric = rankic_2d(pred , label , dim = 0) if label is not None else torch.full_like(pred[0,:] , fill_value=torch.nan)
        return pd.DataFrame({'ic' : ic , 'rankic' : ric} , index = index)

    @staticmethod
    def to_boost_input(data : Any , weight_param : dict , **kwargs) -> BoostInput | Any:
        if data is None: 
            ...
        elif isinstance(data , pd.DataFrame):
            data = BoostInput.from_dataframe(data , weight_param = weight_param , **kwargs)
        elif isinstance(data , np.ndarray):
            data = BoostInput.from_numpy(data[...,:-1] , data[...,-1:] , weight_param = weight_param , **kwargs)
        elif isinstance(data , torch.Tensor):
            data = BoostInput.from_tensor(data[...,:-1] , data[...,-1:] , weight_param = weight_param , **kwargs)
        elif isinstance(data , BoostInput):
            ...
        else:
            raise Exception(data)
        return data
    
    @classmethod
    def df_input(cls , factor_data : pd.DataFrame | None = None , idx : int = -1 , windows_len = 24) -> dict[str,Any]:
        if factor_data is None: 
            factor_data = load_xingye_data()
        MDTs = np.sort(np.unique(factor_data['date'].to_numpy(int)))

        idtEnd = MDTs[idx - 1]
        idtStart = MDTs[idx - windows_len]
        idtTest = MDTs[idx]

        train = factor_data.loc[(factor_data['date'] >= idtStart) & (factor_data['date'] < idtEnd),:].set_index(['date', 'secid']).sort_index()
        valid = factor_data.loc[factor_data['date'] == idtEnd,:].set_index(['date', 'secid']).sort_index()
        test = factor_data.loc[factor_data['date'] == idtTest,:].set_index(['date', 'secid']).sort_index()
        return {'train':train , 'valid' : valid ,  'test':test}
    
    @classmethod
    def mono_constr(cls , train_param : dict , nfeat : int , as_tuple = False):
        raw_mono_constr = train_param.get('monotone_constraints')
        if raw_mono_constr is None or raw_mono_constr == 0: 
            mono_constr = None
        elif isinstance(raw_mono_constr , list):
            if len(raw_mono_constr) == 0 or all(x == 0 for x in raw_mono_constr): 
                mono_constr = None
            else:
                assert len(raw_mono_constr) == nfeat or len(raw_mono_constr) == 1 , (len(raw_mono_constr) , nfeat)
                mono_constr = raw_mono_constr * (nfeat // len(raw_mono_constr))
        else:
            mono_constr = [raw_mono_constr] * nfeat
        if as_tuple and mono_constr is not None: 
            mono_constr = tuple(mono_constr)
        return mono_constr
    
    @property
    def use_gpu(self): 
        return self.cuda and torch.cuda.is_available()

def load_xingye_data():
    factor_data = Load.df(PATH.miscel.joinpath('CombStdByZXMkt_All_TrainLabel.feather')) # 训练集，带Label
    factor_data['date'] = factor_data['date'].astype(str).str.replace('-','').astype(int)
    factor_data['secid'] = DB.code2secid(factor_data['StockID'])

    index_list = ['date','secid']
    label_list = ['nextRtnM']

    factor_data = factor_data.drop(columns=['ZX','mktVal','mktValRank','StockID','nextRtnM_Label']).set_index(index_list)
    factor_rank = factor_data.drop(columns=label_list).groupby('date').rank(pct = True).\
        join(factor_data[label_list]).rename(columns={'nextRtnM':'label'})
    return factor_rank.reset_index().sort_index()