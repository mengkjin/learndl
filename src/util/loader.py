import numpy as np

from torch.utils.data.dataset import IterableDataset , Dataset
from torch.utils.data import Sampler
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

class CustomBatchSampler(Sampler):
    def __init__(self, sampler , batch_size_list , drop_res = True):
        self.sampler = sampler
        self.batch_size_list = np.array(batch_size_list).astype(int)
        assert (self.batch_size_list >= 0).all()
        self.drop_res = drop_res
        
    def __iter__(self):
        if (not self.drop_res) and (sum(self.batch_size_list) < len(self.sampler)):
            new_list = np.append(self.batch_size_list , len(self.sampler) - sum(self.batch_size_list))
        else:
            new_list = self.batch_size_list
        
        batch_count , sample_idx = 0 , 0
        while batch_count < len(new_list):
            if new_list[batch_count] > 0:
                batch = [0] * new_list[batch_count]
                idx_in_batch = 0
                while True:
                    batch[idx_in_batch] = self.sampler[sample_idx]
                    idx_in_batch += 1
                    sample_idx +=1
                    if idx_in_batch == new_list[batch_count]:
                        yield batch
                        break
            batch_count += 1
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self):
        if self.batch_size_list.sum() < len(self.sampler):
            return len(self.batch_size_list) + 1 - self.drop_res
        else:
            return np.where(self.batch_size_list.cumsum() >= len(self.sampler))[0][0] + 1