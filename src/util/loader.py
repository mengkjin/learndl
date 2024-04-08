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
                 mapping = None , shuffle_option : Literal['static' , 'init' , 'epoch'] = 'static',progress_bar = False):
        self.storage  = loader_storage
        self.filelist = batch_file_list
        self.mapping  = self.empty_mapping if mapping is None else mapping
        self.shufopt  = shuffle_option
        self.itertype  = tqdm if progress_bar else list
        if shuffle_option == 'init':
            self.iterator = self.itertype(np.random.permutation(self.filelist))
        else:
            self.iterator = self.itertype(self.filelist)

    def __len__(self):
        return len(self.iterator)
    
    def __iter__(self):
        if self.shufopt == 'epoch':
            self.iterator = self.itertype(np.random.permutation(self.filelist))

        for batch_file in self.iterator: 
            batch_data = self.storage.load(batch_file)
            batch_data = self.mapping(batch_data)
            yield batch_data

    def __getitem__(self , i):
        batch_data = self.mapping(self.storage.load(list(self.iterator)[i]))
        return batch_data
    
    def display(self , text = ''):
        if isinstance(self.iterator, tqdm): self.iterator.set_description(text)

    @staticmethod
    def empty_mapping(x):
        return x

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