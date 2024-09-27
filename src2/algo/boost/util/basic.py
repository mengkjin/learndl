import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from abc import ABC , abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any , Literal , Optional

from .io import BoosterInput , BoosterOutput
from ....basic import PATH
from ....func import ic_2d , rankic_2d

class BasicBoosterModel(ABC):
    DEFAULT_TRAIN_PARAM = {}
    DEFAULT_WEIGHT_PARAM = {}

    def __repr__(self) -> str: return f'{self.__class__.__name__},\n,train_param={self.train_param}'
    
    def __init__(self ,
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 cuda = True , seed = None , **kwargs):   
        self.update_param(train_param , weight_param , cuda = cuda , seed = seed , **kwargs)
        self.data : dict[str,BoosterInput] = {}

    def update_param(self , train_param , weight_param , **kwargs):
        self.train_param = self.new_train_param(train_param , **kwargs)
        self.weight_param = self.new_weight_param(weight_param , **kwargs)
        if 'cuda' in kwargs: self.cuda = kwargs.pop('cuda')
        if 'seed' in kwargs: self.seed = kwargs.pop('seed')
        return self

    def new_train_param(self , train_param , **kwargs) -> dict[str,Any]:
        new_train_param = deepcopy(getattr(self , 'train_param' , self.DEFAULT_TRAIN_PARAM))
        new_train_param.update(train_param)
        return new_train_param

    def new_weight_param(self , weight_param , **kwargs) -> dict[str,Any]:
        new_weight_param = deepcopy(getattr(self , 'weight_param' , self.DEFAULT_WEIGHT_PARAM))
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
    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , **kwargs):
        if train is None: train = self.data['train']
        if valid is None: valid = self.data['valid']
        return self

    def booster_input(self , x : BoosterInput | str | Any = 'test'):
        return self.data[x]  if isinstance(x , str) else x

    @abstractmethod
    def predict(self , x : BoosterInput | str | Any = 'test') -> BoosterOutput:
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
    def load_dict(self , model_dict : dict[str,Any] , cuda = False , seed = None) -> dict[str,Any]: 
        assert self.__class__.__name__ == model_dict['class_name'] , (self.__class__.__name__ , model_dict['class_name'])

        self.cuda = cuda
        self.seed = seed
        self.train_param = model_dict['train_param']
        self.weight_param = model_dict['weight_param']

        return self
    
    @classmethod
    def from_dict(cls , model_dict : dict[str,Any] , cuda = False):
        return cls().load_dict(model_dict , cuda)
    
    def test_result(self , test : BoosterInput | Any = None , plot_path : Path | None = None):
        df = self.calc_ic(test)
        plt.figure()
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(*df.mean().values))
        if plot_path:
            plot_path.mkdir(exist_ok=True)
            plt.savefig(plot_path.joinpath('test_prediction.png'),dpi=1200)
        return df
    
    def calc_ic(self , test : BoosterInput | Any = None):
        if test is None: test = self.data['test']
        label = test.y
        pred  = self.predict(test).to_2d()
        index = test.date

        ic = ic_2d(pred , label , dim = 0)
        ric = rankic_2d(pred , label , dim = 0)
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
    
    @classmethod
    def df_input(cls , factor_data : Optional[pd.DataFrame] = None , idx : int = -1 , windows_len = 24) -> dict[str,Any]:
        if factor_data is None: factor_data = load_xingye_data()
        MDTs = np.sort(factor_data['date'].unique())

        idtEnd = MDTs[idx - 1]
        idtStart = MDTs[idx - windows_len]
        idtTest = MDTs[idx]

        train = factor_data.loc[(factor_data['date'] >= idtStart) & (factor_data['date'] < idtEnd),:].set_index(['date', 'secid']).sort_index()
        valid = factor_data.loc[factor_data['date'] == idtEnd,:].set_index(['date', 'secid']).sort_index()
        test = factor_data.loc[factor_data['date'] == idtTest,:].set_index(['date', 'secid']).sort_index()
        return {'train':train , 'valid' : valid ,  'test':test}

def load_xingye_data():
    factor_data = pd.read_feather(f'{PATH.data}/TreeData/CombStdByZXMkt_All_TrainLabel.feather') # 训练集，带Label
    factor_data['date'] = factor_data['date'].astype(str).str.replace('-','').astype(int)
    v = factor_data['StockID'].astype(str).str.slice(0, 6).replace({'T00018' : '600018'})
    v = v.where(v.str.isdigit() , '-1').astype(int)
    factor_data['secid'] = v

    index_list = ['date','secid']
    label_list = ['nextRtnM']

    factor_data = factor_data.drop(columns=['ZX','mktVal','mktValRank','StockID','nextRtnM_Label']).set_index(index_list)
    factor_rank = factor_data.drop(columns=label_list).groupby('date').rank(pct = True).\
        join(factor_data[label_list]).rename(columns={'nextRtnM':'label'})
    return factor_rank.reset_index().sort_index()