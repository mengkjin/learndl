import torch
import numpy as np

from abc import ABC , abstractmethod
from torch import nn
from typing import Any , Iterator , Literal , Optional

from ...classes import BaseDataModule , BaseTrainerModule , BatchData , BatchOutput , BoosterData
from ...algo.boost.lgbm import Lgbm

class _BaseLgbm(ABC):
    '''load booster data and fit'''
    def __init__(self , model_module : BaseTrainerModule) -> None:
        self.module = model_module
        self.model : Lgbm
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

    @staticmethod
    def batch_data_to_booster_data(net : nn.Module , loader : Iterator[BatchData | Any] , 
                                   secid : Optional[np.ndarray] = None ,
                                   date : Optional[np.ndarray] = None) -> BoosterData:
        '''create booster data of train, valid, test dataloaders'''
        hh , yy , ii = [] , [] , []
        net.eval()
        with torch.no_grad():
            for batch_data in loader:
                hidden = BatchOutput(net(batch_data.x)).hidden
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
        self.model = Lgbm.model_from_string(model_str , cuda=self.is_cuda)
        self.loaded = True
        return self
    @abstractmethod
    def fit(self , *args , **kwargs): '''fit self.model'''
    @abstractmethod
    def predict(self , *args , **kwargs): '''predict certain data'''

class LgbmEnsembler(_BaseLgbm):
    '''load booster data and fit'''
    def booster_data(self , net : nn.Module , loader : Iterator[BatchData | Any]) -> BoosterData:
        return self.batch_data_to_booster_data(net , loader , self.y_secid , self.y_date)

    def fit(self , net : nn.Module):
        net = self.module.device(net)
        train_data = self.booster_data(net , self.train_dl)
        valid_data = self.booster_data(net , self.valid_dl)
        self.model = Lgbm(train_data , valid_data , cuda=self.is_cuda).fit()
        # self.model.plot.training()
        return self
    
    def predict(self):
        assert self.loaded
        hidden = self.module.batch_output.hidden
        if hidden is None: return
        assert isinstance(self.module.batch_data , BatchData)
        hidden = hidden.detach().cpu().numpy()
        label = self.module.batch_data.y.detach().cpu().numpy()
        pred  = torch.tensor(self.model.predict(BoosterData(hidden , label) , reform = False))
        return pred
            
class LgbmBooster(_BaseLgbm):
    '''load booster data and fit'''
    @property
    def train_batch_data(self) -> tuple[BoosterData,BoosterData] | Any:
        assert self.module.status.dataset == 'train' , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , tuple) and len(batch_data) == 2 , batch_data
        assert isinstance(batch_data[0] , BoosterData) and isinstance(batch_data[1] , BoosterData) , batch_data
        return self.module.batch_data
    
    @property
    def test_batch_data(self) -> BoosterData | Any:
        assert self.module.status.dataset == 'test' , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , BoosterData)
        return self.module.batch_data

    def fit(self , *args):
        train_data , valid_data = self.train_batch_data
        self.model = Lgbm(train_data , valid_data , cuda=self.is_cuda).fit()
        # self.model.plot.training()
        return self
    
    def predict(self):
        pred = torch.tensor(self.model.predict(self.test_batch_data , reform = False))
        return pred
            