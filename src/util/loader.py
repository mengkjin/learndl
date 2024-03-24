import gc , os , torch
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data.dataset import IterableDataset , Dataset

class Storage():
    def __init__(self , *args):
        if len(args) > 0 and args[0] in ['disk' , 'mem']:
            self.activate(args[0])

    def activate(self , default = 'mem'):
        assert default in ['disk' , 'mem']
        self.default = default
        self.mem_disk = dict()
        self.file_record = list()
        self.file_group = dict()
    
    def save(self , obj , paths , to_disk = False , group = 'default'):
        for p in self._pathlist(paths): 
            self._saveone(obj , p , self.default == 'disk' or to_disk)
            self._addrecord(p , group)
            
    def load(self , path , from_disk = False):
        if self.default == 'disk' or from_disk:
            return torch.load(path) if os.path.exists(path) else None
        else:
            return self.mem_disk.get(path)

    def _pathlist(self , p):
        if p is None: 
            return []
        elif isinstance(p , str):
            p = [p]
        return p
    
    def _saveone(self , obj , p , to_disk = False):
        if to_disk:
            torch.save(obj , p)
        else:
            self.mem_disk[p] = deepcopy(obj)
    
    def _addrecord(self , p , group):
        self.file_record = np.union1d(self.file_record , p)
        if group not in self.file_group.keys(): 
            self.file_group[group] = [p]
        else:
            self.file_group[group] = np.union1d(self.file_group[group] , [p])
    
    def save_model_state(self , net , paths , to_disk = False , group = 'default'):
        if isinstance(net , torch.nn.Module):
            if self.default == 'disk' or to_disk:
                sd = net.state_dict() 
            else:
                sd = deepcopy(net).cpu().state_dict()
        elif isinstance(net , dict):
            sd = net
        self.save(sd , paths , to_disk , group)
        
    def load_model_state(self , net , path , from_disk = False):
        sd = self.load(path , from_disk)
        net.load_state_dict(sd)
        return net
            
    def valid_paths(self , paths):
        return np.intersect1d(self._pathlist(paths) ,  self.file_record).tolist()
    
    def del_path(self , *args):
        for paths in args:
            if self.default == 'disk':
                [os.remove(p) for p in self._pathlist(paths) if os.path.exists(p)]
            else:
                [self.mem_disk.__delitem__(p) for p in np.intersect1d(self._pathlist(paths) , list(self.mem_disk.keys()))]
            self.file_record = np.setdiff1d(self.file_record , paths)
        gc.collect()
        
    def del_group(self , clear_groups = []):
        for g in self._pathlist(clear_groups):
            paths = self.file_group.get(g)
            if paths is not None:
                self.del_path(paths)
                del self.file_group[g]

class DataloaderStored:
    """
    class of saved dataloader , retrieve batch_data from './model/{model_name}/{set_name}_batch_data'
    """
    def __init__(self, loader_storage : Storage , batch_file_list : list , mapping = None , progress_bar = False):
        self.progress_bar = progress_bar
        self.loader_storage = loader_storage
        self.mapping = mapping
        self.iterator = tqdm(batch_file_list) if progress_bar else batch_file_list
    def __len__(self):
        return len(self.iterator)
    def __iter__(self):
        for batch_file in self.iterator: 
            batch_data = self.loader_storage.load(batch_file)
            if self.mapping is not None:
                batch_data = self.mapping(batch_data)
            yield batch_data
    def __getitem__(self , i):
        batch_data = self.loader_storage.load(list(self.iterator)[i])
        if self.mapping is not None:
            batch_data = self.mapping(batch_data)
        return batch_data
    def display(self , text = ''):
        if isinstance(self.iterator, tqdm): self.iterator.set_description(text)


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