
import torch

from typing import Any , Literal

from .algo import Lgbm , AdaBoost , BasicBoosterModel
from .io import BoosterInput

def choose_booster_model(booster_type):
    if booster_type == 'lgbm':
        return Lgbm
    elif booster_type == 'ada':
        return AdaBoost
    else:
        raise KeyError(booster_type)
    
class GeneralBooster:
    def __init__(self , 
                 booster_type : Literal['ada' , 'lgbm'] | Any = 'lgbm' ,
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 cuda = True , **kwargs):
        self.booster_type = booster_type
        self.train_param = train_param
        self.weight_param = weight_param
        self.cuda = cuda

        self.booster = choose_booster_model(booster_type)(train_param , weight_param , cuda , **kwargs)
        self.booster.import_data(train = train , valid = valid , test = test)

    def __call__(self, x : torch.Tensor): return self.forward(x)

    def fit(self , train = None , valid = None , use_feature = None):
        self.booster.import_data(train = train , valid = valid)
        self.booster.update_feature(use_feature)
        self.booster.fit()
        return self
        
    def predict(self , test : BoosterInput | Any = None):
        return self.booster.predict(test)
    
    def forward(self , x : torch.Tensor) -> torch.Tensor:
        booster_input = BoosterInput.from_tensor(x , feature=self.use_feature)
        return self.booster.predict(booster_input).pred.to(x) 
    
    def to_dict(self):
        return {'booster_type' : self.booster_type , **self.booster.to_dict()}
    
    @property
    def use_feature(self):
        return self.booster.data['train'].use_feature

    @classmethod
    def from_dict(cls , model_dict : dict[str,Any]):
        obj = cls(model_dict['booster_type'])
        obj.booster.load_dict(model_dict)
        return obj
    
    def calc_ic(self , test : BoosterInput | Any = None):
        return self.booster.calc_ic(test)
    
    @classmethod
    def df_input(cls): return BasicBoosterModel.df_input()
