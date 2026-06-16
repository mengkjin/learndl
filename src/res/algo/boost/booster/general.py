"""Unified front-end for all gradient-boost back-ends.

Classes:
    GeneralBoostModel — thin wrapper that dispatches to one of the
                        ``AVAILABLE_BOOSTS`` back-ends and manages parameter
                        splitting between train/weight/verbosity namespaces.
"""
from __future__ import annotations
import torch

from datetime import datetime
from functools import cached_property
from typing import Any

from src.proj.bases.enums import BoostModuleType
from . import ada , catboost , lgbm , xgboost
from ..util import BasicBoostModel , BoostInput

__all__ = ['GeneralBoostModel' , 'AVAILABLE_BOOSTS']

AVAILABLE_BOOSTS = {
    BoostModuleType.ADA : ada.AdaBoost,
    BoostModuleType.LGMB : lgbm.Lgbm,
    BoostModuleType.XGBOOST : xgboost.XgBoost,
    BoostModuleType.CATBOOST : catboost.CatBoost,
}

class GeneralBoostModel:
    """Unified wrapper around LGBM / XGBoost / CatBoost / AdaBoost back-ends.

    Parameter splitting:
        * Keys present in :attr:`BoostWeightMethod.__slots__` go to
          ``weight_param`` and are forwarded to :class:`BoostWeightMethod`.
        * The key ``'verbosity'`` controls :attr:`fit_verbosity` (log frequency).
        * Everything else goes to ``train_param`` for the underlying booster.

    Calling the instance (``model(x)``) is equivalent to :meth:`forward`.
    """
    def __init__(
        self , 
        boost_type : BoostModuleType | str = BoostModuleType.LGMB ,
        params : dict[str, Any] | None = None , * ,
        train : Any = None ,  valid : Any = None , test : Any = None , 
        cuda = True , seed = None , 
        given_name : str | None = None , 
        sub_name : str | None = None ,
        override_criterion : dict | None = None , 
        **kwargs
    ):
        assert boost_type in AVAILABLE_BOOSTS , f'{boost_type} is not a valid boost type'
        self.boost_type = BoostModuleType(boost_type)
        self.given_name = given_name or self.boost.__class__.__name__
        self.sub_name = sub_name or datetime.now().strftime('%Y%m%d-%H%M%S')
        self.override_criterion = override_criterion or {}
        self.cuda = cuda
        self.seed = seed
        self.update_param(params , **kwargs)
        self.import_data(train = train , valid = valid , test = test)

    def __call__(self, x : torch.Tensor) -> torch.Tensor: 
        return self.forward(x)

    def update_param(self , params : dict[str, Any] | None = None , **kwargs):
        self.fit_verbosity = params.get('verbosity' , 10) if params else 10
        self.boost.set_params(params , overrides = self.override_criterion, cuda = self.cuda, seed = self.seed, **kwargs)
        return self

    def import_data(self , train : Any = None , valid : Any = None , test  : Any = None):
        self.boost.import_data(train = train , valid = valid , test = test)
        return self

    def fit(self , train = None , valid = None , use_feature = None , silent = False):
        self.import_data(train = train , valid = valid)
        self.boost.update_feature(use_feature)
        self.boost.fit(silent = silent or self.fit_verbosity < 10)
        return self

    @cached_property
    def boost(self) -> BasicBoostModel:
        return AVAILABLE_BOOSTS[self.boost_type]()
        
    @property
    def data(self):
        return self.boost.data

    def boost_input(self , x : BoostInput | Any = 'test'):
        return self.boost.boost_input(x)
    
    def predict(self , x : BoostInput | Any = 'test'):
        return self.boost.predict(x)
    
    def forward(self , x : torch.Tensor) -> torch.Tensor:
        boost_input = BoostInput.from_tensor(x)
        label = self.boost.predict(boost_input).pred.to(x)         
        return label
    
    def to_dict(self):
        return {'boost_type' : self.boost_type , **self.boost.to_dict()}
    
    def load_dict(self , model_dict , cuda = False , seed = None):
        self.boost.load_dict(model_dict , cuda , seed)
        return self
    
    @property
    def use_feature(self):
        return self.boost.data['train'].use_feature

    @classmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False , seed = None):
        obj = cls(model_dict['boost_type']).load_dict(model_dict , cuda , seed)
        return obj
    
    def calc_ic(self , test : BoostInput | Any = None):
        return self.boost.calc_ic(test)
    
    @classmethod
    def df_input(cls): return BasicBoostModel.df_input()
