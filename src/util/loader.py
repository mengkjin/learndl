import numpy as np

from tqdm import tqdm
from torch.utils.data.dataset import IterableDataset , Dataset
from typing import Literal

from .store import Storage
                
class DataloaderStored:
    """
    class of saved dataloader , retrieve batch_data from Storage
    """
    def __init__(self, loader_storage : Storage , batch_file_list : list , 
                 shuffle_option : Literal['static' , 'init' , 'epoch'] = 'static'):
        self.storage  = loader_storage
        self.shufopt  = shuffle_option
        self.loader   = self.shuf('init' , batch_file_list)

    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        for batch_file in self.shuf('epoch' , self.loader): 
            yield self.storage.load(batch_file)

    def __getitem__(self , i): return self.storage.load(self.loader[i])

    def shuf(self , stage : Literal['init' , 'epoch'] , loader):
        if stage == self.shufopt: loader = np.random.permutation(loader)
        return loader

class Mydataset(Dataset):
    def __init__(self, data1 , label , weight = None) -> None:
            super().__init__()
            self.data1 = data1
            self.label = label
            self.weight = weight
    def __len__(self):
        return len(self.data1)
    def __getitem__(self , ii):
        if self.weight is None:
            return self.data1[ii], self.label[ii]
        else:
            return self.data1[ii], self.label[ii], self.weight[ii]

class MyIterdataset(IterableDataset):
    def __init__(self, data1 , label) -> None:
            super().__init__()
            self.data1 = data1
            self.label = label
    def __len__(self):
        return len(self.data1)
    def __iter__(self):
        for ii in range(len(self.data1)):
            yield self.data1[ii], self.label[ii]