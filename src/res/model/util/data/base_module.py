from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from abc import ABC , abstractmethod
from dataclasses import dataclass
from typing import Any , Iterator , Literal

from src.proj import Logger
from src.data import ModuleData

from src.res.model.util.core import BufferStorage , BatchInput
from src.res.model.util.config import ModelConfig
from .buffer import BaseBuffer

class BaseDataModule(ABC):
    '''A class to store relavant training data'''
    @abstractmethod
    def __init__(self , config : ModelConfig | None = None , use_data : Literal['fit','predict','both'] = 'fit' , *args , **kwargs):
        self.config   : ModelConfig
        self.use_data : Literal['fit','predict','both'] 
        self.storage  : BufferStorage
        self.buffer   : BaseBuffer
    @abstractmethod
    def prepare_data() -> None: 
        '''prepare all data in advance of training'''
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
    @classmethod
    def initialize(cls , config : ModelConfig | None = None , use_data : Literal['fit','predict','both'] = 'fit' , *args , vb_level : Any = 2 , min_key_len = -1 , **kwargs):
        data = cls(config , use_data = use_data , *args , **kwargs)
        Logger.stdout_pairs({'Use Data' : data.use_data} , title = 'Module Data Initiated:' , vb_level = vb_level , min_key_len = min_key_len)
        return data
    def on_before_batch_transfer(self , batch , dataloader_idx = None): return batch
    def transfer_batch_to_device(self , batch , device = None , dataloader_idx = None): return batch
    def on_after_batch_transfer(self , batch , dataloader_idx = None): return batch
    def reset_dataloaders(self):
        '''reset for every fit / test / predict'''
        self.loader_dict  = {}
        self.loader_param = self.LoaderParam()
    def prev_model_date(self , model_date):
        prev_dates = self.model_date_list[self.model_date_list < model_date]
        return max(prev_dates) if len(prev_dates) > 0 else -1
    def next_model_date(self , model_date):
        late_dates = self.model_date_list[self.model_date_list > model_date]
        return min(late_dates) if len(late_dates) > 0 else max(self.test_full_dates) + 1

    @property
    def stage(self) -> Literal['fit' , 'test' , 'predict' , 'extract']:
        return self.loader_param.stage

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
    def min_test_date(self) -> int:
        return self.config.beg_date

    @property
    def max_test_date(self) -> int:
        return self.config.end_date

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

    @dataclass
    class LoaderParam:
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
        
    @property
    def device(self): 
        return self.config.device