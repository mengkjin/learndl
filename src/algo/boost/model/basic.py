import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os , torch

from abc import ABC , abstractmethod
from copy import deepcopy
from typing import Any , Literal

from ..util.io import BoosterInput , BoosterOutput
from ....basic import PATH
from ....func import np_nanic_2d , np_nanrankic_2d

class BasicBoosterModel(ABC):
    DEFAULT_TRAIN_PARAM = {}
    DEFAULT_WEIGHT_PARAM = {}

    def __repr__(self) -> str: return f'{self.__class__.__name__},\n,train_param={self.train_param}'

    def __init__(self ,
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 cuda = True , **kwargs):   
        self.train_param = self.new_train_param(train_param , cuda = cuda , **kwargs)
        self.weight_param = self.new_weight_param(weight_param , cuda = cuda , **kwargs)
        self.cuda = cuda
        self.data : dict[Literal['train','valid','test'],BoosterInput] = {}

    def new_train_param(self , train_param , cuda = True , **kwargs) -> dict[str,Any]:
        new_train_param = deepcopy(self.DEFAULT_TRAIN_PARAM)
        new_train_param.update(train_param)
        return new_train_param

    def new_weight_param(self , weight_param , cuda = True , **kwargs) -> dict[str,Any]:
        new_weight_param = deepcopy(self.DEFAULT_WEIGHT_PARAM)
        new_weight_param.update(weight_param)
        return new_weight_param
    
    def import_data(self , train : Any = None , valid : Any = None , test : Any = None):
        if train is not None: self.data['train'] = self.to_booster_input(train , self.weight_param)
        if valid is not None: self.data['valid'] = self.to_booster_input(valid , self.weight_param)
        if test  is not None: self.data['test']  = self.to_booster_input(test , self.weight_param)
        return self

    def update_feature(self , use_feature = None):
        if use_feature is not None: [bdata.update_feature(use_feature) for bdata in self.data.values()]
        return self

    @abstractmethod
    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None):
        if train is None: train = self.data['train']
        if valid is None: valid = self.data['valid']
        return self
        
    @abstractmethod
    def predict(self , test : BoosterInput) -> BoosterOutput:
        if test is None: test = self.data['test']
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
    def load_dict(self , model_dict : dict[str,Any] , cuda = False) -> dict[str,Any]: 
        assert self.__class__.__name__ == model_dict['class_name'] , (self.__class__.__name__ , model_dict['class_name'])
        self.train_param = model_dict['train_param']
        self.weight_param = model_dict['weight_param']
        self.cuda = cuda
        return self
    
    @classmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False):
        return cls().load_dict(model_dict , cuda)
    
    def test_result(self , test : BoosterInput | Any = None , plot_path : str | None = None):
        df = self.calc_ic(test)
        plt.figure()
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(*df.mean().values))
        if plot_path is not None:
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig('/'.join([plot_path,'test_prediction.png']),dpi=1200)
        return df
    
    def calc_ic(self , test : BoosterInput | Any = None):
        if test is None: test = self.data['test']
        label = test.y
        pred  = self.predict(test).to_2d()
        index = test.date

        ic = np_nanic_2d(pred , label , dim = 0)
        ric = np_nanrankic_2d(pred , label , dim = 0)
        return pd.DataFrame({'ic' : ic , 'rankic' : ric} , index = index)

    @staticmethod
    def to_booster_input(data : Any , weight_param : dict , **kwargs) -> BoosterInput | Any:
        if data is None: 
            ...
        elif isinstance(data , pd.DataFrame):
            data = BoosterInput.from_dataframe(data , weight_param = weight_param , **kwargs)
        elif isinstance(data , np.ndarray):
            data = BoosterInput.from_numpy(data[...,:-1] , data[...,-1:] , weight_param = weight_param , **kwargs)
        elif isinstance(data , torch.Tensor):
            data = BoosterInput.from_tensor(data[...,:-1] , data[...,-1:] , weight_param = weight_param , **kwargs)
        elif isinstance(data , BoosterInput):
            ...
        else:
            raise Exception(data)
        return data
    
    @staticmethod
    def random_input() -> dict[str,Any]:
        def rand_nan(x , ratio = 0.1):
            ii = np.random.choice(np.arange(len(x)) , int(ratio * len(x)))
            x[ii] = np.nan
            return x

        train = rand_nan(np.random.rand(1000,40,20))
        valid = rand_nan(np.random.rand(500,40,20))
        test  = rand_nan(np.random.rand(100,40,20))

        return {'train':train , 'valid':valid , 'test':test}
    
    @staticmethod
    def df_input() -> dict[str,Any]:
        train = pd.read_csv(f'{PATH.data}/TreeData/df_train.csv' , index_col=[0,1])
        valid = pd.read_csv(f'{PATH.data}/TreeData/df_valid.csv' , index_col=[0,1])
        test  = pd.read_csv(f'{PATH.data}/TreeData/df_test.csv' , index_col=[0,1])

        data = {'train':train , 'valid':valid , 'test':test}
        for k , v in data.items():
            v = v.reset_index()

            v['datetime'] = v['datetime'].str.replace('-','').astype(int)
            v['instrument'] = v['instrument'].str.slice(2,8).astype(int)

            data[k] = v.set_index(['datetime','instrument']).sort_index()

        return data
