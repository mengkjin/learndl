import numpy as np
from tqdm import tqdm

from src.proj import Proj
from src.res.model.util import BaseDataModule , BatchInput
    
class BatchInputLoader:
    '''wrap loader to impletement DataModule Callbacks'''
    def __init__(self , raw_loader , data_module : BaseDataModule , exclude_dates = None , include_dates = None , tqdm = True , desc : str | None = None) -> None:
        self.loader      = raw_loader
        self.data_module = data_module
        self.device      = data_module.device
        self.tqdm        = tqdm
        self.desc        = desc
        self.enable_tqdm()
        self.filter_dates(exclude_dates , include_dates)

    def __repr__(self):
        return f'{self.__class__.__name__}(length={len(self.loader)})'

    def __len__(self):  return len(self.loader)
    def __getitem__(self , i : int): return self.process(list(self.loader)[i] , i)
    def __iter__(self):
        for batch_i , batch_input in enumerate(self.loader):
            assert isinstance(batch_input , BatchInput) , f'{type(batch_input)} is not a BatchInput'
            if self.exclude_dates is not None or self.include_dates is not None:
                batch_date  = self.data_module.batch_date0(batch_input)
                if self.exclude_dates is not None and np.isin(batch_date , self.exclude_dates): 
                    continue
                if self.include_dates is not None and ~np.isin(batch_date , self.include_dates): 
                    continue
            yield self.process(batch_input , batch_i)        
    
    def process(self , batch_input : BatchInput , batch_i : int) -> BatchInput:
        batch_input = self.data_module.on_before_batch_transfer(batch_input , batch_i)
        batch_input = self.data_module.transfer_batch_to_device(batch_input , self.device , batch_i)
        batch_input = self.data_module.on_after_batch_transfer(batch_input , batch_i)
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

    def of_date(self , date : int):
        assert self.data_module.config.sample_method == 'sequential' or self.data_module.stage != 'fit' , (self.data_module.config.sample_method , self.data_module.stage)
        for batch_input in self:
            if self.data_module.batch_date(batch_input)[0] == date:
                return batch_input
        raise ValueError(f'date {date} not found in loader')
