import numpy as np
from tqdm import tqdm

from src.proj import SILENT
from src.res.model.util import BaseDataModule , BatchData
    
class BatchDataLoader:
    '''wrap loader to impletement DataModule Callbacks'''
    def __init__(self , raw_loader , data_module : BaseDataModule , exclude_dates = None , include_dates = None , tqdm = True) -> None:
        self.loader      = raw_loader
        self.data_module = data_module
        self.device      = data_module.device
        self.verbosity   = data_module.config.verbosity
        if self.verbosity >= 10 and tqdm: 
            self.enable_tqdm()
        self.filter_dates(exclude_dates , include_dates)

    def __repr__(self):
        return f'{self.__class__.__name__}(length={len(self.loader)})'

    def __len__(self):  return len(self.loader)
    def __getitem__(self , i : int): return self.process(list(self.loader)[i] , i)
    def __iter__(self):
        for batch_i , batch_data in enumerate(self.loader):
            assert isinstance(batch_data , BatchData) , f'{type(batch_data)} is not a BatchData'
            if self.exclude_dates is not None or self.include_dates is not None:
                batch_date  = self.data_module.batch_date0(batch_data)
                if self.exclude_dates is not None and np.isin(batch_date , self.exclude_dates): 
                    continue
                if self.include_dates is not None and ~np.isin(batch_date , self.include_dates): 
                    continue
            yield self.process(batch_data , batch_i)        
    
    def process(self , batch_data : BatchData , batch_i : int) -> BatchData:
        batch_data = self.data_module.on_before_batch_transfer(batch_data , batch_i)
        batch_data = self.data_module.transfer_batch_to_device(batch_data , self.device , batch_i)
        batch_data = self.data_module.on_after_batch_transfer(batch_data , batch_i)
        return batch_data

    def enable_tqdm(self , disable = False):
        if not isinstance(self.loader , tqdm): 
            self.loader = tqdm(self.loader , total=len(self.loader))
        self.loader.disable = disable or SILENT
        return self

    def display(self , text : str):
        if isinstance(self.loader , tqdm) and not self.loader.disable:  
            self.loader.set_description(text)

    def filter_dates(self , exclude_dates = None , include_dates = None):
        if exclude_dates is not None or include_dates is not None:
            assert self.data_module.config.train_sample_method == 'sequential' , self.data_module.config.train_sample_method
        self.exclude_dates = exclude_dates
        self.include_dates = include_dates
        return self

    def of_date(self , date : int):
        assert self.data_module.config.train_sample_method == 'sequential' or self.data_module.stage != 'fit' , (self.data_module.config.train_sample_method , self.data_module.stage)
        for batch_data in self:
            if self.data_module.batch_date(batch_data)[0] == date:
                return batch_data
        raise ValueError(f'date {date} not found in loader')
