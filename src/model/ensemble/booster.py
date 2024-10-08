import torch
import numpy as np

from torch import nn , Tensor
from typing import Any , Iterator , Literal , Optional

from ..classes import BaseDataModule , BaseTrainer , BatchData , BatchOutput , BoosterInput , GeneralBooster , TrainConfig

class BoosterModel:
    '''Booster used in nn_model'''
    def __init__(self , model_module : BaseTrainer) -> None:
        self.module = model_module
        self.model : GeneralBooster

    def __bool__(self): return True
    @property
    def config(self) -> TrainConfig: return self.module.config
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
    def model_dict(self): return self.model.to_dict()
    @property
    def is_cuda(self) -> bool: return self.module.device.device.type == 'cuda'
    @property
    def booster_params(self): 
        if self.config.model_booster_head:
            return self.config.booster_head_param
        else:
            return self.module.model_param

    @property
    def batch_data(self) -> BatchData:
        assert isinstance(self.module.batch_data , BatchData)
        return self.module.batch_data
    
    def reset(self): self.loaded = False
    def load(self , model_dict : dict):
        '''load self.model'''
        self.model = GeneralBooster.from_dict(model_dict , cuda = self.is_cuda , seed = self.config.random_seed)
        self.loaded = True
        return self
    
    def booster_input(self , dataset : Literal['train' , 'valid' , 'test'] = 'test' , *args , **kwargs):
        if dataset == 'train':   booster_input = self.train_booster_input(*args , **kwargs)
        elif dataset == 'valid': booster_input = self.valid_booster_input(*args , **kwargs)
        elif dataset == 'test':  booster_input = self.test_booster_input(*args , **kwargs)
        return booster_input

    def train_booster_input(self , *args , **kwargs) -> BoosterInput | Any:
        assert self.module.status.dataset in ['train','valid'] , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , tuple) and len(batch_data) == 2 and isinstance(batch_data[0] , BoosterInput) , batch_data
        return batch_data[0]
    
    def valid_booster_input(self , *args , **kwargs) -> BoosterInput | Any:
        assert self.module.status.dataset in ['train','valid'] , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , tuple) and len(batch_data) == 2 and isinstance(batch_data[1] , BoosterInput) , batch_data
        return batch_data[1]
    
    def test_booster_input(self , *args , **kwargs) -> BoosterInput | Any:
        assert self.module.status.dataset == 'test' , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , BoosterInput)
        return batch_data

    def fit(self , *args , silence = False , **kwargs):
        train_data = self.booster_input('train', *args , **kwargs)
        valid_data = self.booster_input('valid', *args , **kwargs)
        if silence: self.booster_params['verbosity'] = 0
        self.model = GeneralBooster(self.config.booster_type , self.booster_params , 
                                    train = train_data , valid = valid_data , 
                                    cuda=self.is_cuda , seed = self.config.random_seed).fit()
        self.loaded = True
        return self
    
    def predict(self , dataset : Literal['train' , 'valid' , 'test'] = 'test' , *args , **kwargs):
        booster_input = self.booster_input(dataset , *args , **kwargs)
        pred  = self.model.predict(booster_input).to_2d()
        output = BatchOutput((pred , {'label' : booster_input.y}))
        return output
    
class BoosterEnsembler(BoosterModel):
    '''BoosterEnsembler used in nn_model'''
    def train_booster_input(self , net : nn.Module , *args , **kwargs) -> BoosterInput | Any: 
        return self.loader_to_booster_input(net , self.train_dl)
    
    def valid_booster_input(self , net : nn.Module , *args , **kwargs) -> BoosterInput | Any:
        return self.loader_to_booster_input(net , self.valid_dl)
    
    def test_booster_input(self , *args ,  **kwargs) -> BoosterInput | Any:
        '''test datas , usually by date'''
        assert self.loaded
        hidden : Tensor = self.module.batch_output.other['hidden']
        return BoosterInput.from_tensor(hidden , self.batch_data.y)
    
    @staticmethod
    def batch_data_to_booster_input(net : nn.Module , loader : Iterator[BatchData | Any] , 
                                    secid : Optional[np.ndarray] = None ,
                                    date : Optional[np.ndarray] = None) -> BoosterInput:
        '''create booster data of train, valid, test dataloaders'''
        hh , yy , ii = [] , [] , []
        net.eval()
        with torch.no_grad():
            for batch_data in loader:
                hidden : Tensor = BatchOutput(net(batch_data.x)).other['hidden']
                assert hidden is not None , f'hidden must not be none when using BoosterModel'
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
        return BoosterInput.from_numpy(hh_values , yy_values , secid , date)
    
    def loader_to_booster_input(self , net : nn.Module , loader : Iterator[BatchData | Any]) -> BoosterInput:
        return self.batch_data_to_booster_input(net , loader , self.y_secid , self.y_date)
