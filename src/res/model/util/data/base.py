from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from abc import ABC , abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any , Iterator , Literal , Callable

from src.proj import Logger , Proj , CALENDAR , Const
from src.data import ModuleData

from src.res.model.util.core import BatchInput
from src.res.model.util.config import ModelConfig
from src.res.model.util.storage import TorchFileStorage , StoredTorchFileLoader
from .dynamic_buffer import DynamicDataBuffer

__all__ = ['BaseDataModule', 'DataloaderParam']

@dataclass
class DataloaderParam:
    stage : Literal['fit' , 'test' , 'predict' , 'extract'] | Any = None
    model_date : int | Any = None
    seqlens : dict[str,int] | Any = None
    extract_backward_days : int | Any = None
    extract_forward_days  : int | Any = None

    def __post_init__(self):
        assert self.stage is None or self.stage in ['fit' , 'test' , 'predict' , 'extract'] , self.stage
        assert self.model_date is None or self.model_date > 0 , self.model_date
        assert self.seqlens is None or self.seqlens , self.seqlens
        if self.seqlens is None:
            self.seqlens = {}
        if self.stage != 'extract':
            self.extract_backward_days = None 
            self.extract_forward_days  = None

class BaseDataModule(ABC):
    '''A class to store relavant training data'''
   
    def __init__(self , config : ModelConfig | None = None , use_data : Literal['fit','predict','both'] = 'fit' , *args , vb_level : Any = 1 , **kwargs):
        self.config   : ModelConfig = config or ModelConfig(stage=0)
        self.use_data : Literal['fit','predict','both'] = use_data
        self._vb_level = Proj.vb(vb_level)

    def __repr__(self): 
        keys =  self.input_keys
        if len(keys) >= 5: 
            keys_str = f'[{keys[0]},...,{keys[-1]}({len(keys)})]'
        else:
            keys_str = str(keys)
        return f'{self.__class__.__name__}(model_name={self.config.model_name},use_data={self.use_data},datas={keys_str})'    
    
    @abstractmethod
    def load_data(self) -> None: 
        '''load prepared data at training begin , only load data once in a fitting'''
        self.model_date_list : np.ndarray
        self.test_full_dates : np.ndarray
        self.datas : ModuleData
        self.labels : torch.Tensor
    @abstractmethod
    def setup(self , *args , **kwargs) -> None: 
        '''create train / valid / test dataloaders , perform in every different model_date / model_num'''
        self.d0 : int
        self.d1 : int
        self.y_std : torch.Tensor
        self.early_test_dates : np.ndarray
        self.model_test_dates : np.ndarray
    @abstractmethod
    def train_dataloader(self)  -> Iterator[BatchInput]: 
        '''return train dataloaders'''
    @abstractmethod
    def val_dataloader(self)    -> Iterator[BatchInput]: 
        '''return valid dataloaders'''
    @abstractmethod
    def test_dataloader(self)   -> Iterator[BatchInput]: 
        '''return test dataloaders'''
    @abstractmethod
    def predict_dataloader(self)-> Iterator[BatchInput]: 
        '''return predict dataloaders'''
    @property
    def callback_dict(self) -> dict[str,list[Callable]]:
        if not hasattr(self , '_callback_dict'):
            self._callback_dict = defaultdict(list)
        return self._callback_dict
    def register_callbacks(self , hook_name : str , *callbacks : Callable):
        assert hook_name in ['on_before_batch_transfer' , 'on_after_batch_transfer'] , hook_name
        for callback in callbacks:
            self.callback_dict[hook_name].append(callback)
    
    def print_out(self , vb_level : Any = 2 , min_key_len = 30):
        Logger.stdout_pairs({'Use Data' : self.use_data} , title = 'Module Data Initiated:' , vb_level = vb_level , min_key_len = min_key_len)
    def on_before_batch_transfer(self , batch : BatchInput) -> BatchInput: 
        for callback in self.callback_dict['on_before_batch_transfer']:
            batch = callback(batch)
        return batch
    def on_after_batch_transfer(self , batch : BatchInput) -> BatchInput: 
        for callback in self.callback_dict['on_after_batch_transfer']:
            batch = callback(batch)
        return batch
    def transfer_batch_to_device(self , batch : BatchInput , device = None):
        if self.config.module_type == 'nn':
            batch = batch.to(self.config.device if device is None else device)
        return batch
    
    def reset_dataloaders(self):
        '''reset for every fit / test / predict'''
        self.loader_dict : dict[str , StoredTorchFileLoader]  = {}
        self.loader_dates : dict[str , list[int]] = {}
        self.loader_param = DataloaderParam()

    def empty_dataloader(self) -> None:
        if self.is_fitting:
            self.loader_dict['train'] = StoredTorchFileLoader(self.storage , [] , 'static')
            self.loader_dict['valid'] = StoredTorchFileLoader(self.storage , [] , 'static')
        else:
            self.loader_dict[self.stage] = StoredTorchFileLoader(self.storage , [] , 'static')
            self.loader_dates[self.stage] = []

    def prev_model_date(self , model_date):
        prev_dates = self.model_date_list[self.model_date_list < model_date]
        return max(prev_dates) if len(prev_dates) > 0 else -1
    def next_model_date(self , model_date):
        late_dates = self.model_date_list[self.model_date_list > model_date]
        return min(late_dates) if len(late_dates) > 0 else max(self.test_full_dates) + 1

    @property
    def vb_level(self) -> int:
        return self._vb_level
    def stdout(self , *args , add_vb : int = 0 , **kwargs):
        if 'vb_level' not in kwargs:
            kwargs['vb_level'] = self.vb_level + add_vb
        Logger.stdout(f'{self.__class__.__name__} :' , *args , **kwargs)
    def alert1(self , *args , **kwargs):
        if 'vb_level' not in kwargs:
            kwargs['vb_level'] = self.vb_level
        Logger.alert1(f'{self.__class__.__name__}' , *args , **kwargs)

    @property
    def stage(self) -> Literal['fit' , 'test' , 'predict' , 'extract']:
        return self.loader_param.stage

    @property
    def storage(self):
        if not hasattr(self , '_storage'):
            self._storage = TorchFileStorage(self.config.mem_storage)
        return self._storage

    @property
    def buffer(self):
        if not hasattr(self , '_buffer'):
            self._buffer = DynamicDataBuffer(self.config.device)
        return self._buffer

    @property
    def is_fitting(self): return self.stage == 'fit'

    @property
    def input_keys(self) -> list[str]:
        input_keys = [key for value in self.config.input_keys_all.values() for key in value]
        if self.config.module_type == 'factor':
            input_keys.append('factor')
        return input_keys

    @property
    def input_keys_all(self) -> dict[str,list[str]]:
        input_keys = {key : [*value] for key , value in self.config.input_keys_all.items()}
        if self.config.module_type == 'factor':
            input_keys['factor'] = ['factor']
        input_keys = {key : value for key , value in input_keys.items() if value}
        assert len(input_keys) > 0 , (self.config.input_keys_all , self.config.module_type)
        return input_keys

    @property
    def input_keys_subkeys(self) -> dict[str,str]:
        try:
            subkeys = {f'{key}.{subkey}' : str(list(self.datas.x[subkey].feature)) for key , value in self.input_keys_all.items() for subkey in value if subkey in self.datas.x}
        except Exception as e:
            Logger.alert2(f'Error getting input keys subkeys: {e}')
            Logger.alert2(f'Input keys: {self.input_keys}')
            return {f'{key}.{subkey}' : subkey for key , value in self.input_keys_all.items() for subkey in value}
        return subkeys

    @property
    def model_date(self) -> int:
        return self.loader_param.model_date

    @property
    def seq_lens(self) -> dict[str,int]:
        return self.loader_param.seqlens

    @property
    def y_secid(self) -> np.ndarray:
        return self.datas.y.secid

    @property
    def y_date(self) -> np.ndarray:
        return self.datas.y.date[self.d0:self.d1]

    @property
    def day_len(self) -> int:
        return self.d1 - self.d0

    @property
    def data_step(self) -> int:
        return self.config.fitting_step if self.stage in ['fit'] else 1

    @property
    def empty_x(self):
        return self.datas.empty_x and not self.input_keys_hidden

    @property
    def beg_date(self):
        return 19000101 if self.use_data == 'predict' else self.config.beg_date

    @property
    def end_date(self):
        return 99991231 if self.use_data == 'predict' else self.config.end_date

    @property
    def input_type(self) -> Literal['data' , 'hidden' , 'factor' , 'combo']: 
        return self.config.input_type

    @property
    def input_keys_data(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.input_data_types]

    @property
    def input_keys_factor(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.input_factor_types]

    @property
    def input_keys_hidden(self) -> list[str]:
        return [ModuleData.abbr(key) for key in self.config.input_hidden_types]

    @property
    def seq_steps(self) -> dict[str,int]:
        return self.config.seq_steps

    @property
    def min_test_date(self):
        if hasattr(self , 'test_full_dates'):
            return self.test_full_dates.min() if len(self.test_full_dates) > 0 else 99991231
        return ModuleData.min_data_date(self.input_keys_data + self.input_keys_factor , factor_names = self.config.input_factor_names) or 99991231

    @property
    def max_test_date(self):
        if hasattr(self , 'test_full_dates'):
            return self.test_full_dates.max() if len(self.test_full_dates) > 0 else 19000101
        return ModuleData.max_data_date(self.input_keys_data + self.input_keys_factor , factor_names = self.config.input_factor_names) or 19000101

    @property
    def factor_start_dt(self):
        beg_date = self.beg_date
        if self.config.is_null_model and self.config.is_resuming and Const.Model.resume_test:
            beg_date = max(beg_date , self.config.resumed_max_pred_date)
        return CALENDAR.td(beg_date , -1).as_int()

    @property
    def factor_end_dt(self):
        return self.end_date
     

    def y_label(self , dates : np.ndarray | list[int]) -> pd.DataFrame:
        labels : list[pd.DataFrame] = []
        for date in dates:
            label = self.label_of_date(date)
            if label.size > 0:
                labels.append(pd.DataFrame({
                    'secid' : self.datas.y.secid, 'date' : date,
                    'label' : label.flatten()
                }).dropna())
        return pd.concat(labels)

    def label_of_date(self , date : int) -> np.ndarray:
        return self.labels[:,self.datas.y.date == date][...,0].squeeze().cpu().numpy()