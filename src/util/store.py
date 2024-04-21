import gc , os , torch
import numpy as np
import pandas as pd

from copy import deepcopy

from ..environ import DIR

class Storage:
    def __init__(self , mem_storage : bool = False):
        self.memdisk = {} if mem_storage else None
        self.records = pd.DataFrame(columns = ['path' , 'group'] , dtype = str)

    @property
    def is_disk(self): return self.memdisk is None
    @property
    def is_mem(self): return not self.is_disk
    
    def exists(self , path):
        if self.memdisk is None:
            return os.path.exists(path)
        else:
            return path in self.memdisk.keys()

    def save(self , obj , path , group = 'default'):
        if isinstance(path , (list , tuple)):
            [self.save(obj , p , group = group) for p in path]
        elif isinstance(path , str):
            if self.memdisk is None:
                torch.save(obj , path)
            else:
                # assert not self.is_cuda(obj)
                self.memdisk[path] = deepcopy(obj)
            df = pd.DataFrame({'path' : [path] , 'group' : [group]})
            self.records = pd.concat([self.records[self.records['path'] != path] , df] , axis=0)
        else:
            raise TypeError(type(path))
        
    def is_cuda(self , obj) -> bool:
        if isinstance(obj , (torch.Tensor , torch.nn.Module)):
            return obj.is_cuda
        elif isinstance(obj , (list , tuple)):
            for sub in obj:
                if self.is_cuda(sub): return True
            else:
                return False
        elif isinstance(obj , dict):
            return self.is_cuda(list(obj.values()))
        else:
            return False

    def load(self , path):
        if isinstance(path , (list , tuple)):
            return [self.load(p) for p in path]
        elif isinstance(path , str):
            if self.memdisk is None:
                return torch.load(path) if os.path.exists(path) else None
            else:
                return self.memdisk.get(path)
        else:
            raise TypeError(type(path))
    
    def save_state_dict(self , obj , path , group = 'default'):
        assert isinstance(obj , (torch.nn.Module , dict)) , obj
        if isinstance(obj , torch.nn.Module):
            obj = deepcopy(obj).cpu().state_dict()
        else:
            obj = deepcopy(obj)
            [val.detach().cpu() for val in obj.values() if isinstance(val , torch.Tensor)]
        self.save(obj , path , group)
        
    def load_state_dict(self , obj , path):
        sd = self.load(path)
        assert isinstance(obj , torch.nn.Module) , obj
        assert isinstance(sd , dict) , sd
        obj.load_state_dict(sd)
        return obj
            
    def valid_path(self , path):
        if isinstance(path , str): path = [path]
        return np.intersect1d(path , self.records['path']).tolist()
    
    def del_path(self , path):
        path = self.valid_path(path)
        if self.memdisk is None:
            [os.remove(p) for p in path]
        else:
            [self.memdisk.__delitem__(p) for p in path]
        self.records = self.records[~self.records['path'].isin(path)]

    def del_all(self):
        self.del_path(self.records['path'])
        gc.collect()
        
    def del_group(self , group):
        if isinstance(group , str): group = [group]
        path = self.records['path'][self.records['group'].isin(group)]
        self.del_path(path)
        gc.collect()

    