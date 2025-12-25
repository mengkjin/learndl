import torch

from typing import Any , Literal

from . import ada , catboost , lgbm , xgboost
from ..util import BasicBoosterModel , BoosterInput , BoosterWeightMethod

AVAILABLE_BOOSTERS = {
    'lgbm' : lgbm.Lgbm,
    'ada' : ada.AdaBoost,
    'xgboost' : xgboost.XgBoost,
    'catboost' : catboost.CatBoost,
}

class GeneralBooster:
    def __init__(self , 
                 booster_type : Literal['ada' , 'lgbm' , 'xgboost' , 'catboost'] | str = 'lgbm' ,
                 params : dict[str,Any] = {} ,
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 cuda = True , seed = None , given_name : str | None = None , **kwargs):
        assert booster_type in AVAILABLE_BOOSTERS , f'{booster_type} is not a valid booster type'
        self.booster_type = booster_type
        self.booster : BasicBoosterModel = AVAILABLE_BOOSTERS[self.booster_type]()
        self.given_name = given_name
        self.update_param(params , cuda = cuda , seed = seed , **kwargs)
        self.import_data(train = train , valid = valid , test = test)

    def __call__(self, x : torch.Tensor) -> torch.Tensor: 
        return self.forward(x)

    def update_param(self , params : dict[str,Any] , **kwargs):
        self.train_param = {k:v for k,v in params.items() if k not in [*BoosterWeightMethod.__slots__,'verbosity','seqlens']}
        self.weight_param = {k:v for k,v in params.items() if k in BoosterWeightMethod.__slots__}
        self.fit_verbosity = params.get('verbosity' , 10)

        self.booster.update_param(self.train_param , self.weight_param , **kwargs)
        return self

    def import_data(self , train : Any = None , valid : Any = None , test  : Any = None):
        self.booster.import_data(train = train , valid = valid , test = test)
        return self

    def fit(self , train = None , valid = None , use_feature = None , silent = False):
        self.import_data(train = train , valid = valid)
        self.booster.update_feature(use_feature)
        self.booster.fit(silent = silent or self.fit_verbosity < 10)
        return self
        
    @property
    def data(self):
        return self.booster.data

    def booster_input(self , x : BoosterInput | str | Any = 'test'):
        return self.booster.booster_input(x)
    
    def predict(self , x : BoosterInput | str | Any = 'test'):
        return self.booster.predict(x)
    
    def forward(self , x : torch.Tensor) -> torch.Tensor:
        booster_input = BoosterInput.from_tensor(x)
        return self.booster.predict(booster_input).pred.to(x) 
    
    def to_dict(self):
        return {'booster_type' : self.booster_type , **self.booster.to_dict()}
    
    def load_dict(self , model_dict , cuda = False , seed = None):
        self.booster.load_dict(model_dict , cuda , seed)
        return self
    
    @property
    def use_feature(self):
        return self.booster.data['train'].use_feature

    @classmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False , seed = None):
        obj = cls(model_dict['booster_type']).load_dict(model_dict , cuda , seed)
        return obj
    
    def calc_ic(self , test : BoosterInput | Any = None):
        return self.booster.calc_ic(test)
    
    @classmethod
    def df_input(cls): return BasicBoosterModel.df_input()
