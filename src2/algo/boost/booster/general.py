
import torch

from typing import Any , Literal

from .ada import AdaBoost
from .catboost import CatBoost
from .lgbm import Lgbm
from .xgboost import XgBoost

from ..util.io import BoosterInput , BoosterWeightMethod
from ..util.basic import BasicBoosterModel

VALID_BOOSTERS = ['lgbm' , 'ada' , 'xgboost' , 'catboost']
_booster_dict = {
    'lgbm' : Lgbm,
    'ada' : AdaBoost,
    'xgboost' : XgBoost,
    'catboost' : CatBoost,
}

def choose_booster_model(booster_type : str):
    return _booster_dict[booster_type]

class GeneralBooster:
    def __init__(self , 
                 booster_type : Literal['ada' , 'lgbm'] | Any = 'lgbm' ,
                 params : dict[str,Any] = {} ,
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 cuda = True , seed = None , **kwargs):
        assert booster_type , booster_type
        self.booster_type = booster_type
        self.train_param = {k:v for k,v in params.items() if k not in BoosterWeightMethod.__slots__}
        self.weight_param = {k:v for k,v in params.items() if k in BoosterWeightMethod.__slots__}
        self.cuda = cuda
        self.seed = seed
        self.verbosity = params.get('verbosity' , 10)

        self.booster : BasicBoosterModel = choose_booster_model(booster_type)(self.train_param , self.weight_param , cuda , seed , **kwargs)
        self.import_data(train = train , valid = valid , test = test)

    def __call__(self, x : torch.Tensor): return self.forward(x)

    def import_data(self , train : Any = None , valid : Any = None , test  : Any = None):
        self.booster.import_data(train , valid , test)

    def fit(self , train = None , valid = None , use_feature = None):
        self.import_data(train = train , valid = valid)
        self.booster.update_feature(use_feature)
        self.booster.fit(silence = self.verbosity < 10)
        return self
        
    def predict(self , test : BoosterInput | Any = None):
        return self.booster.predict(test)
    
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
