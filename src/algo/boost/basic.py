import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os , torch

from abc import ABC , abstractmethod , abstractclassmethod

from typing import Any , Optional

from ...basic import PATH
from ...data import BoosterData
from ...func import np_nanic_2d , np_nanrankic_2d

plt.style.use('seaborn-v0_8') 

class BasicBooster(ABC):
    def __init__(self , 
                 train : Any = None , 
                 valid : Any = None ,
                 test  : Any = None , 
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 feature = None , 
                 plot_path = None, # '../../figures' ,
                 cuda = True , **kwargs):   
        self.train_param = train_param
        self.weight_param = weight_param
        self.plot_path = plot_path
        self.data : dict[str , BoosterData] = {}
        self.feature = feature
        self.cuda = cuda
        self.data_import(train = train , valid = valid , test = test)

    def data_import(self , train : Any = None , valid : Any = None , test : Any = None):
        if train is not None: self.data['train'] = self.data_transform(train)
        if valid is not None: self.data['valid'] = self.data_transform(valid)
        if test  is not None: self.data['test']  = self.data_transform(test)
        return self

    def data_transform(self , data : Any) -> BoosterData:
        if isinstance(data , pd.DataFrame):
            data = BoosterData(data.iloc[:,:-1] , data.iloc[:,-1] , feature = self.feature , **self.weight_param)
        elif isinstance(data , (np.ndarray , torch.Tensor)):
            data = BoosterData(data[...,:-1] , data[...,-1] , feature = self.feature , **self.weight_param)
        elif isinstance(data , BoosterData):
            ...
        else:
            raise Exception(data)
        return data

    @abstractmethod
    def fit(self , use_feature = None , train = None , valid = None ):
        ...
        
    @abstractmethod
    def predict(self , inputs : Optional[BoosterData] = None , reshape = True , reform = True):
       ...
    
    @abstractmethod
    def to_dict(self) -> dict[str,Any]: ...
    
    @classmethod
    @abstractmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False):
        return cls()

    @property
    def initiated(self): return bool(self.data)

    def test_result(self):
        df = self.calc_ic()
        plt.figure()
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(*df.mean().values))
        if self.plot_path is not None:
            os.makedirs(self.plot_path, exist_ok=True)
            plt.savefig('/'.join([self.plot_path,'test_prediction.png']),dpi=1200)
        return df
    
    def calc_ic(self , pred : np.ndarray | None = None , label : np.ndarray | None = None , index = None):
        if label is None:
            label = self.data['test'].y
        if pred is None:
            pred = np.array(self.predict(self.data['test'], reshape = True , reform = False)).reshape(*label.shape)
        if index is None:
            index = self.data['test'].date
        if pred.ndim == 1: pred , label = pred.reshape(-1,1) , label.reshape(-1,1)
        ic = np_nanic_2d(pred , label , dim = 0)
        ric = np_nanrankic_2d(pred , label , dim = 0)
        return pd.DataFrame({'ic' : ic , 'rankic' : ric} , index = index)

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
        train = pd.read_csv(f'{PATH.data}/tree_data/df_train.csv' , index_col=[0,1])
        valid = pd.read_csv(f'{PATH.data}/tree_data/df_valid.csv' , index_col=[0,1])
        test  = pd.read_csv(f'{PATH.data}/tree_data/df_test.csv' , index_col=[0,1])

        return {'train':train , 'valid':valid , 'test':test}