import gc , os , torch
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Any , Literal

from .config import TrainConfig

class Storage:
    '''Interface of mem or disk storage, methods'''
    def __init__(self , store_type : Literal['mem' , 'disk'] = 'disk'):
        self.memdisk = {} if store_type == 'mem' else None
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
    
    def del_path(self , path):
        if isinstance(path , str): path = [path]
        path = np.intersect1d(path , self.records['path']).tolist()
        if self.memdisk is None:
            [os.remove(p) for p in path]
        else:
            [self.memdisk.__delitem__(p) for p in path]
        self.records = self.records[~self.records['path'].isin(path)]
        
    def del_group(self , group):
        if isinstance(group , str): group = [group]
        path = self.records['path'][self.records['group'].isin(group)]
        self.del_path(path)
        gc.collect()

    def del_all(self):
        self.del_path(self.records['path'])
        gc.collect()

class Checkpoint(Storage):
    '''model check point for epochs'''
    def __init__(self, store_type: Literal['mem' , 'disk'] | TrainConfig):
        if isinstance(store_type , TrainConfig): store_type = 'mem' if store_type.mem_storage else 'disk'
        super().__init__(store_type)
        self.epoch_queue : list[list] = []
        self.join_record : list[str]  = [] 

    def new_model(self , model_param : dict , model_date : int):
        path = (os.path.basename if self.is_mem else str)(str(model_param.get('path')))
        self.dir = '{}/{}'.format(path , model_date)
        self.epoch_queue = []
        self.join_record = [] 
        self.del_all()
    
    def join(self , src : Any , epoch : int , net):
        if epoch < 0: return
        if epoch >= len(self.epoch_queue): self._extend_reliance()
        record_str = f'JOIN: Epoch {epoch}, from {src.__class__}({id(src)})'
        if src in self.epoch_queue[epoch]: 
            record_str += ', already exists'
        else:
            self.epoch_queue[epoch].append(src)
            record_str += ', append list'
        if self.exists(self.epoch_path(epoch)): 
            record_str += ', no need to save'
        else:
            self.save_state_dict(net , self.epoch_path(epoch))
            record_str += ', state dict saved'
        self.join_record.append(record_str)

    def disjoin(self , src , epoch : int):
        if epoch < 0: return
        record_str = f'DISJOIN: Epoch {epoch}, from {src.__class__}({id(src)})'
        self.epoch_queue[epoch] = [s for s in self.epoch_queue[epoch] if s is not src]
        if self.epoch_queue[epoch]:
            record_str += f', {len(self.epoch_queue[epoch])} reliance left'
        else:
            self.del_path(self.epoch_path(epoch))
            record_str += f', delete state dict '
        self.join_record.append(record_str)

    def load_epoch(self , epoch):
        assert epoch >= 0 , epoch
        if self.epoch_queue[epoch]:
            return self.load(self.epoch_path(epoch))
        else:
            [print(record_str) for record_str in self.join_record]
            raise Exception(f'no checkpoint of epoch {epoch}')
    
    def epoch_path(self , epoch): return f'{self.dir}/checkpoint.{epoch}.pt'
    
    def _extend_reliance(self , n_epochs = 200):
        self.epoch_queue += [[] for _ in range(n_epochs)] # extend epoch list

class Deposition(Storage):
    '''model saver'''
    def __init__(self , store_type : Any = None):
        super().__init__('disk')

    