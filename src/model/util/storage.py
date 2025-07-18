import gc , torch
import numpy as np
import pandas as pd

from copy import deepcopy
from pathlib import Path
from typing import Any , Literal

from src.basic import ModelDict , ModelPath , torch_load

class MemFileStorage:
    '''Interface of mem or disk storage, methods'''
    def __init__(self , mem_storage : bool = False):
        self._mem_storage = mem_storage
        self.memdisk = {} if mem_storage else None
        self.records = pd.DataFrame(columns = ['path' , 'group'] , dtype = str)

    @property
    def is_disk(self): return not self.is_disk
    @property
    def is_mem(self): return self._mem_storage
    
    def exists(self , path):
        if isinstance(path , str): path = Path(path)
        return path.exists() if self.memdisk is None else (str(path) in self.memdisk.keys())

    def save(self , obj , path , group = 'default'):
        if isinstance(path , list):
            [self.save(obj , p , group = group) for p in path]
        elif isinstance(path , (Path , str)):
            if self.memdisk is None:
                torch.save(obj , path)
            else:
                self.memdisk[str(path)] = deepcopy(obj)
            df = pd.DataFrame({'path' : [str(path)] , 'group' : [group]})
            self.records = pd.concat([self.records[self.records['path'] != str(path)] , df] , axis=0)
        else:
            raise TypeError(type(path))

    def load(self , path) -> Any:
        if isinstance(path , list):
            return [self.load(p) for p in path]
        elif isinstance(path , (str | Path)):
            if self.memdisk is None:
                return torch_load(path) if Path(path).exists() else None
            else:
                return self.memdisk.get(str(path))
        else:
            raise TypeError(type(path))
    
    def is_cuda(self , obj) -> bool:
        if isinstance(obj , (torch.Tensor , torch.nn.Module)):
            return bool(obj.is_cuda)
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
        self.save(obj , path , group)
    
    def del_path(self , path):
        if isinstance(path , (Path , str)): path = [path]
        if self.memdisk is None:
            [Path(p).unlink() for p in path]
        else:
            [self.memdisk.__delitem__(str(p)) for p in path]
        self.records = self.records[~self.records['path'].isin([str(p) for p in path])]
        
    def del_group(self , group):
        if isinstance(group , str): group = [group]
        path = self.records['path'][self.records['group'].isin(group)]
        self.del_path(path)
        gc.collect()

    def del_all(self):
        self.del_path(self.records['path'])
        gc.collect()

class StoredFileLoader:
    ''''retrieve batch_data from a Storage'''
    def __init__(self, loader_storage : MemFileStorage , file_list : list , 
                 shuffle_option : Literal['static' , 'init' , 'epoch'] = 'static'):
        self.storage  = loader_storage
        self.shufopt  = shuffle_option
        self.file_loader   = self.shuf('init' , file_list)

    def __len__(self): return len(self.file_loader)
    def __getitem__(self , i): return self.storage.load(self.file_loader[i])
    def __iter__(self):
        for batch_file in self.shuf('epoch' , self.file_loader): 
            yield self.storage.load(batch_file)
    def shuf(self , stage : Literal['init' , 'epoch'] , loader):
        '''shuffle at init or each epoch'''
        if stage == self.shufopt: loader = np.random.permutation(loader)
        return loader
    
class Checkpoint(MemFileStorage):
    '''model checkpoint for epochs'''
    def __init__(self, mem_storage: bool):
        super().__init__(mem_storage)
        self.epoch_queue : list[list] = []
        self.join_record : list[str]  = [] 
        # self.model_module = model_module

    def new_model(self , model_param : dict , model_date : int):
        path = Path(model_param.get('path' , ''))
        if self.is_mem:
            self.dir = f'{path.name}/{model_date}'
        else:
            self.dir = path.joinpath(str(model_date))
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
        
        if epoch >= len(self.epoch_queue):
            [print(record_str) for record_str in self.join_record]
            
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
    
    def epoch_path(self , epoch): 
        if isinstance(self.dir , Path):
            return self.dir.joinpath(f'checkpoint.{epoch}.pt')
        else:
            return f'{self.dir}/checkpoint.{epoch}.pt'
    
    def _extend_reliance(self , n_epochs = 200):
        self.epoch_queue += [[] for _ in range(n_epochs)] # extend epoch list

class Deposition:
    '''model saver'''
    def __init__(self , base_path : ModelPath):
        self.base_path = base_path

    def stack_model(self , model_dict : ModelDict , model_num , model_date , submodel = 'best'):
        model_dict.save(self.model_path(model_num , model_date , submodel) , stack = True)

    def dump_model(self , model_num , model_date , submodel = 'best'):
        model_path = self.model_path(model_num , model_date , submodel)
        for path in model_path.iterdir():
            if path.stem.endswith('.stack'):
                new_path = path.with_stem(path.stem.replace('.stack',''))
                if new_path.exists(): new_path.unlink()
                path.rename(new_path)

    def load_model(self , model_num , model_date , submodel = 'best'):
        return self.base_path.model_file(model_num , model_date , submodel)
    
    def exists(self , model_num , model_date , submodel = 'best'):
        return self.base_path.exists(model_num , model_date , submodel)
    
    def model_path(self , model_num , model_date , submodel = 'best'):
        '''get model path of deposition giving model date / num / submodel'''
        return self.base_path.full_path(model_num , model_date , submodel)