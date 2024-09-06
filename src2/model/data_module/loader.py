from tqdm import tqdm

from ..util import BatchData
from ..util.classes import BaseDataModule
    
class BatchDataLoader:
    '''wrap loader to impletement DataModule Callbacks'''
    def __init__(self , raw_loader , data_module : BaseDataModule) -> None:
        self.loader      = raw_loader
        self.data_module = data_module
        self.device      = data_module.device
        self.verbosity   = data_module.config.verbosity
        if self.verbosity >= 10: self.init_tqdm()

    def __len__(self):  return len(self.loader)
    def __getitem__(self , i : int): return self.process(list(self.loader)[i] , i)
    def __iter__(self):
        for batch_i , batch_data in enumerate(self.loader):
            yield self.process(batch_data , batch_i)        
    
    def process(self , batch_data : BatchData , batch_i : int) -> BatchData:
        batch_data = self.data_module.on_before_batch_transfer(batch_data , batch_i)
        batch_data = self.data_module.transfer_batch_to_device(batch_data , self.device , batch_i)
        batch_data = self.data_module.on_after_batch_transfer(batch_data , batch_i)
        return batch_data

    def init_tqdm(self):
        self.loader = tqdm(self.loader , total=len(self.loader))

    def display(self , text : str):
        if isinstance(self.loader , tqdm):  self.loader.set_description(text)