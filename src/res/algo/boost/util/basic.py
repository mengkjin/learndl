import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from abc import ABC , abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any , Optional

from src.proj import PATH
from src.basic import DB
from src.func import ic_2d , rankic_2d

from .io import BoosterInput , BoosterOutput

__all__ = ['BasicBoosterModel' , 'load_xingye_data']

class BasicBoosterModel(ABC):
    DEFAULT_TRAIN_PARAM = {}
    DEFAULT_WEIGHT_PARAM = {}
    DEFAULT_CATEGORICAL_N_BINS = 3
    DEFAULT_CATEGORICAL_MAX_BINS = 10

    def __repr__(self) -> str: return f'{self.__class__.__name__},\n,train_param={self.train_param}'
    
    def __init__(self ,
                 train_param : dict[str,Any] = {} ,
                 weight_param : dict[str,Any] = {} ,
                 cuda = True , seed = None , **kwargs):   
        self.train_param = deepcopy(self.DEFAULT_TRAIN_PARAM)
        self.weight_param = deepcopy(self.DEFAULT_WEIGHT_PARAM)
        self.cuda = cuda
        self.seed = seed
        self.update_param(train_param , weight_param , **kwargs)
        self.data : dict[str,BoosterInput] = {}

    def update_param(self , train_param , weight_param , **kwargs):
        self.train_param.update(train_param) 
        self.weight_param.update(weight_param)
        if 'cuda' in kwargs: 
            self.cuda = kwargs.pop('cuda')
        if 'seed' in kwargs: 
            self.seed = kwargs.pop('seed')
        self.assert_param()
        return self
    
    def assert_param(self): 
        assert all(k in self.DEFAULT_TRAIN_PARAM for k in self.train_param) , \
            f'{str([k for k in self.train_param if k not in self.DEFAULT_TRAIN_PARAM])} not in DEFAULT_TRAIN_PARAM'
    
    def import_data(self , train : Any = None , valid : Any = None , test : Any = None):
        if train is not None: 
            self.data['train'] = self.to_booster_input(train , self.weight_param)
        if valid is not None: 
            self.data['valid'] = self.to_booster_input(valid , self.weight_param)
        if test  is not None: 
            self.data['test']  = self.to_booster_input(test , self.weight_param)
        return self

    def update_feature(self , use_feature = None):
        if use_feature is not None: 
            [bdata.update_feature(use_feature) for bdata in self.data.values()]
        return self

    @abstractmethod
    def fit(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silent = False , **kwargs):
        self.booster_fit_inputs(train , valid , silent)
        return self
    
    @property
    def booster_objective_multi(self):
        obj = self.train_param.get('objective' , None)
        return obj is not None and 'softmax' in obj
    
    @property
    def booster_objective_rank(self):
        obj = self.train_param.get('objective' , None)
        return obj is not None and 'rank' in obj

    def booster_input(self , x : BoosterInput | str | Any = 'test'):
        return self.data[x] if isinstance(x , str) else x

    def booster_fit_inputs(self , train : BoosterInput | Any = None , valid : BoosterInput | Any = None , silent = False , **kwargs):
        self.silent = silent

        train_param = {k:v for k,v in deepcopy(self.train_param).items() if k in self.DEFAULT_TRAIN_PARAM}

        # categorical_label
        n_bins = train_param.pop('n_bins', None)
        if self.booster_objective_multi:
            if n_bins is None: 
                n_bins = self.DEFAULT_CATEGORICAL_N_BINS
                if not self.silent: 
                    print(f'n_bins not specified, using default value {self.DEFAULT_CATEGORICAL_N_BINS}')
            n_bins = min(n_bins , self.DEFAULT_CATEGORICAL_MAX_BINS)            
        train_param['n_bins'] = n_bins

        if train is None: 
            train = self.data['train']
        if valid is None: 
            valid = self.data['valid']

        self.fit_train_ds = train.to_categorical(n_bins).Dataset()
        self.fit_valid_ds = valid.to_categorical(n_bins).Dataset()
        self.fit_train_param = train_param

        return self
    
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
    
    def test_result(self , test : BoosterInput | Any = None , plot_path : Path | None = None):
        df = self.calc_ic(test)
        plt.figure()
        
        df.cumsum().plot(title='average IC/RankIC = {:.4f}/{:.4f}'.format(df['ic'].mean() , df['rankic'].mean()))
        if plot_path:
            plot_path.mkdir(exist_ok=True)
            plt.savefig(plot_path.joinpath('test_prediction.png'),dpi=1200)
        return df
    
    def calc_ic(self , test : BoosterInput | Any = None):
        if test is None: 
            test = self.data['test']
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
        if factor_data is None: 
            factor_data = load_xingye_data()
        MDTs = np.sort(np.unique(factor_data['date'].to_numpy()))

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
    factor_data = pd.read_feather(f'{PATH.miscel}/CombStdByZXMkt_All_TrainLabel.feather') # 训练集，带Label
    factor_data['date'] = factor_data['date'].astype(str).str.replace('-','').astype(int)
    factor_data['secid'] = DB.code_to_secid(factor_data['StockID'])

    index_list = ['date','secid']
    label_list = ['nextRtnM']

    factor_data = factor_data.drop(columns=['ZX','mktVal','mktValRank','StockID','nextRtnM_Label']).set_index(index_list)
    factor_rank = factor_data.drop(columns=label_list).groupby('date').rank(pct = True).\
        join(factor_data[label_list]).rename(columns={'nextRtnM':'label'})
    return factor_rank.reset_index().sort_index()