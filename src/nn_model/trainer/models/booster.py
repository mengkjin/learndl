import torch
import numpy as np

from abc import ABC , abstractmethod
from torch import nn , Tensor
from typing import Any , Iterator , Literal , Optional

from ...classes import BaseDataModule , BaseTrainer , BatchData , BatchOutput , BoosterData
from ....algo.boost.lgbm import Lgbm as algo_lgbm

def choose_booster(module):
    if module == 'lgbm': return LgbmBooster
    elif module == 'hidden_aggregator': return LgbmBooster
    else: raise KeyError(module)

class Booster(ABC):
    @property
    def train_batch_data(self) -> tuple[BoosterData,BoosterData] | Any: 
        '''train & valid datas , usually altogether'''
        return (None , None)
    @property
    def test_batch_data(self) -> BoosterData | Any:
        '''test datas , usually by date'''
        return None
    @abstractmethod
    def reset(self):
        self.model : Any
        self.loaded = False
        '''reset loader boooter'''
        return self
    @abstractmethod
    def load(self ,  *args , **kwarg):
        '''load booster from anything'''
        return self
    @abstractmethod
    def fit(self , *args , **kwargs): 
        '''fit self.model'''
        return self
    @abstractmethod
    def predict(self , *args , **kwargs) -> Tensor: 
        '''predict certain data based on self.model'''
    @abstractmethod
    def label(self , *args , **kwargs) -> Tensor: 
        '''return dataset label based on input'''

class LGBM(Booster):
    '''Light GBM'''
    def __init__(self , model_module : BaseTrainer) -> None:
        self.module = model_module
        self.model : algo_lgbm

    def __bool__(self): return True
    @property
    def data(self) -> BaseDataModule: return self.module.data
    @property
    def train_dl(self): return self.data.train_dataloader()
    @property
    def valid_dl(self): return self.data.val_dataloader()
    @property
    def test_dl(self): return self.data.test_dataloader()
    @property
    def predcit_dl(self): return self.data.predict_dataloader()
    @property
    def y_secid(self) -> np.ndarray: return self.data.y_secid
    @property
    def y_date(self) -> np.ndarray: return self.data.y_date
    @property
    def model_string(self): return self.model.model_to_string()
    @property
    def is_cuda(self) -> bool: return self.module.device.device.type == 'cuda'
    @property
    def lgbm_params(self): return {'seed' : self.module.config['random_seed'] , **self.module.model_param}

    @staticmethod
    def batch_data_to_booster_data(net : nn.Module , loader : Iterator[BatchData | Any] , 
                                   secid : Optional[np.ndarray] = None ,
                                   date : Optional[np.ndarray] = None) -> BoosterData:
        '''create booster data of train, valid, test dataloaders'''
        hh , yy , ii = [] , [] , []
        net.eval()
        with torch.no_grad():
            for batch_data in loader:
                hidden : Tensor = BatchOutput(net(batch_data.x)).other['hidden']
                assert hidden is not None , f'hidden must not be none when using LgbmEnsembler'
                hh.append(hidden.detach().cpu())
                yy.append(batch_data.y.detach().cpu())
                ii.append(batch_data.i.detach().cpu())
        hh , yy , ii = torch.vstack(hh).numpy() , torch.vstack(yy).numpy() , torch.vstack(ii).numpy()
        secid_i , secid_j = np.unique(ii[:,0] , return_inverse=True)
        date_i  , date_j  = np.unique(ii[:,1] , return_inverse=True)
        hh_values = np.full((len(secid_i) , len(date_i) , hh.shape[-1]) , fill_value = np.nan)
        yy_values = np.full((len(secid_i) , len(date_i)) , fill_value = np.nan)
        
        hh_values[secid_j , date_j] = hh[:]
        yy_values[secid_j , date_j] = yy[...,0]
        secid = secid[secid_i] if secid is not None else None
        date  = date[date_i]   if date  is not None else None
        return BoosterData(hh_values , yy_values , secid , date)
    
    def reset(self): self.loaded = False
    def load(self , model_str: str):
        '''load self.model'''
        self.model = algo_lgbm.model_from_string(model_str , cuda=self.is_cuda)
        self.loaded = True
        return self

            
class LgbmBooster(LGBM):
    '''load booster data and fit'''
    @property
    def train_batch_data(self) -> tuple[BoosterData,BoosterData] | Any:
        assert self.module.status.dataset in ['train','valid'] , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , tuple) and len(batch_data) == 2 , batch_data
        assert isinstance(batch_data[0] , BoosterData) and isinstance(batch_data[1] , BoosterData) , batch_data
        return batch_data
    
    @property
    def test_batch_data(self) -> BoosterData | Any:
        assert self.module.status.dataset == 'test' , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , BoosterData)
        return batch_data

    def fit(self , *args):
        train_data , valid_data = self.train_batch_data
        self.model = algo_lgbm(train_data , valid_data , cuda=self.is_cuda , **self.lgbm_params).fit()
        # self.model.plot.training()
        return self
    
    def predict(self , dataset : Literal['train' , 'valid' , 'test'] = 'test'):
        if dataset == 'train':
            x = self.train_batch_data[0]
        elif dataset == 'valid': 
            x = self.train_batch_data[1]
        else: 
            x = self.test_batch_data
        return torch.tensor(self.model.predict(x , reform = False))
    
    def label(self , dataset : Literal['train' , 'valid' , 'test'] = 'test'):
        if dataset == 'train': 
            return torch.tensor(self.train_batch_data[0].Y())
        elif dataset == 'valid': 
            return torch.tensor(self.train_batch_data[1].Y())
        else: 
            return torch.tensor(self.test_batch_data.Y())
