import torch

from typing import Any , Literal

from . import ada , catboost , lgbm , xgboost
from ..util import BasicBoostModel , BoostInput , BoostWeightMethod

AVAILABLE_BOOSTS = {
    'lgbm' : lgbm.Lgbm,
    'ada' : ada.AdaBoost,
    'xgboost' : xgboost.XgBoost,
    'catboost' : catboost.CatBoost,
}

class GeneralBoostModel:
    def __init__(self , 
                 boost_type : Literal['ada' , 'lgbm' , 'xgboost' , 'catboost'] | str = 'lgbm' ,
                 params : dict[str,Any] = {} ,
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 cuda = True , seed = None , given_name : str | None = None , **kwargs):
        assert boost_type in AVAILABLE_BOOSTS , f'{boost_type} is not a valid boost type'
        self.boost_type = boost_type
        self.boost : BasicBoostModel = AVAILABLE_BOOSTS[self.boost_type]()
        self.given_name = given_name
        self.update_param(params , cuda = cuda , seed = seed , **kwargs)
        self.import_data(train = train , valid = valid , test = test)

    def __call__(self, x : torch.Tensor) -> torch.Tensor: 
        return self.forward(x)

    def update_param(self , params : dict[str,Any] , **kwargs):
        self.train_param = {k:v for k,v in params.items() if k not in [*BoostWeightMethod.__slots__,'verbosity','seqlens']}
        self.weight_param = {k:v for k,v in params.items() if k in BoostWeightMethod.__slots__}
        self.fit_verbosity = params.get('verbosity' , 10)

        self.boost.update_param(self.train_param , self.weight_param , **kwargs)
        return self

    def import_data(self , train : Any = None , valid : Any = None , test  : Any = None):
        self.boost.import_data(train = train , valid = valid , test = test)
        return self

    def fit(self , train = None , valid = None , use_feature = None , silent = False):
        self.import_data(train = train , valid = valid)
        self.boost.update_feature(use_feature)
        self.boost.fit(silent = silent or self.fit_verbosity < 10)
        return self
        
    @property
    def data(self):
        return self.boost.data

    def boost_input(self , x : BoostInput | str | Any = 'test'):
        return self.boost.boost_input(x)
    
    def predict(self , x : BoostInput | str | Any = 'test'):
        return self.boost.predict(x)
    
    def forward(self , x : torch.Tensor) -> torch.Tensor:
        boost_input = BoostInput.from_tensor(x)
        return self.boost.predict(boost_input).pred.to(x) 
    
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
