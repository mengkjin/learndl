import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os , torch

from abc import ABC , abstractmethod , abstractclassmethod
from copy import deepcopy
from typing import Any , Optional

from .util.data import BoosterData , BoosterOutput
from ...basic import PATH
from ...func import np_nanic_2d , np_nanrankic_2d

plt.style.use('seaborn-v0_8') 

class BasicBooster(ABC):
    DEFAULT_TRAIN_PARAM = {}
    DEFAULT_WEIGHT_PARAM = {
        'ts_type' : None ,
        'cs_type' : None ,
        'bm_type' : None , 
        'ts_lin_rate' : 0.5 ,
        'ts_half_life_rate' : 0.5 ,
        'cs_top_tau' : 0.75*np.log(0.5)/np.log(0.75) ,
        'cs_ones_rate' : 2. ,
        'bm_rate' : 2. ,
        'bm_secid' : None}

    def __init__(self ,
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 plot_path = None, # '../../figures' ,
                 cuda = True , **kwargs):   
        self.train_param = self.new_train_param(train_param , cuda = cuda , **kwargs)
        self.weight_param = self.new_weight_param(weight_param , cuda = cuda , **kwargs)
        self.plot_path = plot_path
        self.cuda = cuda
        self.data_import(train = train , valid = valid , test = test)

    def data_import(self , train : Any = None , valid : Any = None , test : Any = None):
        if train is not None: self.train_data = self.to_booster_data(train , self.weight_param)
        if valid is not None: self.valid_data = self.to_booster_data(valid , self.weight_param)
        if test  is not None: self.test_data  = self.to_booster_data(test , self.weight_param)
        return self

    def update_feature(self , use_feature = None):
        if use_feature is None and self.train_data: use_feature = self.train_data.use_feature
        if self.train_data: self.train_data.update_feature(use_feature)
        if self.valid_data: self.valid_data.update_feature(use_feature)
        if self.test_data : self.test_data.update_feature(use_feature)
        return self

    def new_train_param(self , train_param , cuda = True , **kwargs) -> dict[str,Any]:
        new_train_param = deepcopy(self.DEFAULT_TRAIN_PARAM)
        new_train_param.update(train_param)
        return new_train_param

    def new_weight_param(self , weight_param , cuda = True , **kwargs) -> dict[str,Any]:
        new_weight_param = deepcopy(self.DEFAULT_WEIGHT_PARAM)
        new_weight_param.update(weight_param)
        return new_weight_param

    @abstractmethod
    def fit(self , use_feature = None , train = None , valid = None):
        self.model : Any
        return self
        
    @abstractmethod
    def predict(self , inputs : BoosterData | Any = None , reshape = True , as_dataframe = False):
        if inputs is None: inputs = self.test_data
        pred = np.array(self.model.predict(inputs.X()))
        pred = inputs.output_reform(pred , reshape , as_dataframe)
        return pred
    
    @abstractmethod
    def to_dict(self) -> dict[str,Any]: ...
    
    @classmethod
    @abstractmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False):
        return cls()

    def test_result(self):
        df = self.calc_ic()
        plt.figure()
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(*df.mean().values))
        if self.plot_path is not None:
            os.makedirs(self.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.plot_path,'test_prediction.png']),dpi=1200)
        return df
    
    def calc_ic(self , pred : np.ndarray | Any = None , label : np.ndarray | Any = None , index = None):
        if label is None:
            label = self.test_data.y
        if pred is None:
            pred = np.array(self.predict(self.test_data, reshape = True)).reshape(*label.shape)
        if index is None:
            index = self.test_data.date
        if pred.ndim == 1: pred , label = pred.reshape(-1,1) , label.reshape(-1,1)
        ic = np_nanic_2d(pred , label , dim = 0)
        ric = np_nanrankic_2d(pred , label , dim = 0)
        return pd.DataFrame({'ic' : ic , 'rankic' : ric} , index = index)

    @staticmethod
    def to_booster_data(data : Any , weight_param : dict , **kwargs) -> BoosterData | Any:
        if data is None: 
            ...
        elif isinstance(data , pd.DataFrame):
            data = BoosterData.from_dataframe(data , weight_param = weight_param , **kwargs)
        elif isinstance(data , np.ndarray):
            data = BoosterData.from_numpy(data[...,:-1] , data[...,-1:] , weight_param = weight_param , **kwargs)
        elif isinstance(data , torch.Tensor):
            data = BoosterData.from_tensor(data[...,:-1] , data[...,-1:] , weight_param = weight_param , **kwargs)
        elif isinstance(data , BoosterData):
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