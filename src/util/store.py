import gc , os , torch
import numpy as np
import pandas as pd

from copy import deepcopy

class Storage:
    def __init__(self , mem_storage : bool = True):
        self.mem_storage = mem_storage
        self.mem_disk = dict()
        self.file_records = pd.DataFrame(columns = ['path' , 'group'] , dtype = str)
    
    def save(self , obj , paths , to_disk = False , group = 'default'):
        [self.insert_one(obj , p , group , (not self.mem_storage) or to_disk) for p in self.path_list(paths)]
            
    def load(self , path , from_disk = False):
        if (not self.mem_storage) or from_disk:
            return torch.load(path) if os.path.exists(path) else None
        else:
            return self.mem_disk.get(path)
    
    def insert_one(self , obj , p , group , to_disk):
        if to_disk:
            torch.save(obj , p)
        else:
            self.mem_disk[p] = deepcopy(obj)
        df = pd.DataFrame({'path' : [p] , 'group' : [group]})
        self.file_records = pd.concat([self.file_records , df],axis=0)
    
    def save_model_state(self , net , paths , to_disk = False , group = 'default'):
        assert isinstance(net , (torch.nn.Module , dict)) , net
        disk = (not self.mem_storage) or to_disk
        if isinstance(net , torch.nn.Module) and disk:
            sd = net.state_dict() 
        elif isinstance(net , torch.nn.Module):
            sd = deepcopy(net).cpu().state_dict()
        elif disk:
            sd = net
        else:
            sd = deepcopy(net)
        self.save(sd , paths , to_disk , group)
        
    def load_model_state(self , net , path , from_disk = False):
        sd = self.load(path , from_disk)
        net.load_state_dict(sd)
        return net
            
    def valid_paths(self , paths):
        return np.intersect1d(self.path_list(paths) , self.file_records['path']).tolist()
    
    def del_path(self , paths):
        paths = self.valid_paths(paths)
        if self.mem_storage:
            [self.mem_disk.__delitem__(path) for path in paths if path in self.mem_disk.keys()]
        else:
            [os.remove(path) for path in paths if os.path.exists(path)]
        self.file_records = self.file_records[~self.file_records['path'].isin(paths)]
        gc.collect()

    def del_all(self):
        self.del_path(self.file_records['path'])
        
    def del_group(self , groups):
        if isinstance(groups , str): groups = [groups]
        paths = self.file_records['path'][self.file_records['group'].isin(groups)]
        self.del_path(paths)

    @staticmethod
    def path_list(p : str | list | tuple | None):
        if p is None: 
            return []
        elif isinstance(p , str):
            p = [p]
        return p
    