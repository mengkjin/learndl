import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from typing import Any , Literal , Sequence

from src.proj import Proj
from src.res.model.util.core import BatchInput

__all__ = ['BatchInputLoader' , 'DataloaderParam']

@dataclass
class DataloaderParam:
    stage : Literal['fit' , 'test' , 'predict' , 'extract' , 'retrospective'] = 'fit'
    model_date : int | Any = None
    seqlens : dict[str,int] | Any = None
    extract_backward_days : int | Any = None
    extract_forward_days  : int | Any = None
    retro_start_date : int | Any = None
    retro_end_date : int | Any = None

    def __post_init__(self):
        assert self.stage in ['fit' , 'test' , 'predict' , 'extract' , 'retrospective'] , self.stage
        if self.stage == 'retrospective':
            if self.retro_start_date is None:
                self.retro_start_date = self.model_date // 10000 * 10000 + 101
            if self.retro_end_date is None:
                self.retro_end_date = self.model_date // 10000 * 10000 + 1231
            assert self.retro_start_date <= self.retro_end_date , (self.retro_start_date , self.retro_end_date)
        assert self.model_date is None or self.model_date > 0 , self.model_date
        assert self.seqlens is None or self.seqlens , self.seqlens
        if self.seqlens is None:
            self.seqlens = {}
        if self.stage != 'extract':
            self.extract_backward_days = None 
            self.extract_forward_days  = None

    def __eq__(self , other : Any) -> bool:
        assert isinstance(other , DataloaderParam) , f'must compare with instance of DataloaderParam, but got {type(other)}'
        if self.stage != other.stage:
            return False
        match_attrs = ['seqlens']
        if self.stage == 'extract':
            match_attrs.extend(['model_date' , 'extract_backward_days' , 'extract_forward_days'])
        elif self.stage == 'retrospective':
            match_attrs.extend(['retro_start_date' , 'retro_end_date'])
        else:
            match_attrs.append('model_date')
        return all([getattr(self , attr) == getattr(other , attr) for attr in match_attrs])
class BatchInputLoader:
    '''wrap loader to impletement DataModule Callbacks'''
    def __init__(self , raw_loader : Sequence[BatchInput] , data_module , exclude_dates = None , include_dates = None , tqdm = True , desc : str | None = None) -> None:
        from src.res.model.util.data import DataModule
        assert isinstance(data_module , DataModule) , f'data_module must be an instance of BaseDataModule, but got {type(data_module)}'
        self.loader      = raw_loader
        self.data_module = data_module
        self.tqdm        = tqdm
        self.desc        = desc
        self.enable_tqdm()
        self.filter_dates(exclude_dates , include_dates)

    @property
    def device(self):
        return self.data_module.config.device

    def __repr__(self):
        return f'{self.__class__.__name__}(length={len(self.loader)})'

    def __len__(self):  
        return len(self.loader)
    def __getitem__(self , i : int) -> BatchInput: 
        if i < 0:
            i += len(self)
        assert 0 <= i < len(self) , f'index {i} is out of range {len(self)}'
        for idx , batch_input in enumerate(self.loader):
            if idx == i:
                return self.process(batch_input)
        raise IndexError(f'index {i} is out of range {len(self)}')
    def __iter__(self):
        for batch_input in self.loader:
            assert isinstance(batch_input , BatchInput) , f'{type(batch_input)} is not a BatchInput'
            if self.exclude_dates is not None or self.include_dates is not None:
                batch_date  = batch_input.date0
                if self.exclude_dates is not None and np.isin(batch_date , self.exclude_dates): 
                    continue
                if self.include_dates is not None and ~np.isin(batch_date , self.include_dates): 
                    continue
            yield self.process(batch_input)        
    
    def process(self , batch_input : BatchInput) -> BatchInput:
        batch_input = self.data_module.on_before_batch_transfer(batch_input)
        batch_input = self.data_module.transfer_batch_to_device(batch_input , self.device)
        batch_input = self.data_module.on_after_batch_transfer(batch_input)
        return batch_input

    def enable_tqdm(self , disable = False):
        if not Proj.vb.is_max_level or not self.tqdm or disable: 
            return self
        if not isinstance(self.loader , tqdm): 
            self.loader = tqdm(self.loader , total=len(self.loader) , desc=self.desc)
        return self

    def display(self , text : str):
        if isinstance(self.loader , tqdm) and not self.loader.disable:  
            self.loader.set_description(text.rstrip())

    def filter_dates(self , exclude_dates = None , include_dates = None):
        if exclude_dates is not None or include_dates is not None:
            assert self.data_module.config.sample_method == 'sequential' , self.data_module.config.sample_method
        self.exclude_dates = exclude_dates
        self.include_dates = include_dates
        return self

    @property
    def batch_dates(self) -> list[int]:
        return [batch_input.date0 for batch_input in self.loader]

    def of_date(self , date : int):
        assert self.data_module.config.sample_method == 'sequential' or self.data_module.stage != 'fit' , (self.data_module.config.sample_method , self.data_module.stage)
        for batch_input in self.loader:
            if batch_input.date0 == date:
                return self.process(batch_input)
        raise ValueError(f'date {date} not found in loader of dates {self.batch_dates}')
