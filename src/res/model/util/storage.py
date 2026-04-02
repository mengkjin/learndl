from __future__ import annotations

import gc , torch
import numpy as np
import pandas as pd

from copy import deepcopy
from pathlib import Path
from typing import Any , Literal , TypeVar

from src.proj import Logger , PATH
from src.proj.util import torch_load
from .model_path import ModelDict , ModelPath

T = TypeVar('T')
class MemFileStorage:
    '''Interface of mem or disk storage, methods'''
    def __init__(self , mem_storage : bool = False):
        self.is_mem = mem_storage
        self.memdisk : dict[str,Any] = {}
        self.records = pd.DataFrame(columns = pd.Index(['key' , 'path' , 'group']) , dtype = str)

    @property
    def keys(self): 
        return list(self.memdisk.keys())

    def real_path(self , key : str): 
        return PATH.batch.joinpath(f'{key}.pt')
    
    def exists(self , key : str):
        if self.is_mem:
            return key in self.memdisk.keys()
        else:
            return self.real_path(key).exists()

    def save(self , obj : Any , key : str , group = 'default'):
        if self.is_mem:
            path = key
            self.memdisk[key] = deepcopy(obj)
        else:
            path = self.real_path(key)
            torch.save(obj , path)
            self.memdisk[key] = 1
            
        df = pd.DataFrame({'path' : [str(path)] , 'key' : [key] , 'group' : [group]})
        self.records = pd.concat([self.records.query('key != @key') , df] , axis=0)

    def load(self , key : str) -> Any:
        if self.is_mem:
            return self.memdisk[key]
        else:
            path = self.real_path(key)
            return torch_load(path) if path.exists() else None
        
    def save_state_dict(self , obj : Any , key : str , group = 'default'):
        assert isinstance(obj , (torch.nn.Module , dict)) , obj
        if isinstance(obj , torch.nn.Module):
            obj = deepcopy(obj).cpu().state_dict()
        else:
            obj = deepcopy(obj)
        self.save(obj , key , group)
    
    def del_one(self , key : str):
        if not self.is_mem:
            self.real_path(key).unlink(missing_ok=True)
        self.memdisk.pop(key)
        self.records = self.records.query('key != @key')
        
    def del_group(self , group : str):

        for key in self.records.query('group != @group')['key']:
            self.del_one(key)

    def del_all(self):
        for key in self.records['key']:
            self.del_one(key)
        gc.collect()

class StoredFileLoader:
    ''''retrieve batch_input from a Storage'''
    def __init__(self, loader_storage : MemFileStorage , keys : list[str] , shuffle_option : Literal['static' , 'init' , 'epoch'] = 'static'):
        self.storage = loader_storage
        self.shufopt = shuffle_option
        self.keys   = self.shuf(keys)

    def __len__(self): 
        return len(self.keys)
    def __getitem__(self , i): 
        return self.storage.load(self.keys[i])
    def __iter__(self):
        for key in self.shuf(self.keys , 'epoch'): 
            yield self.storage.load(key)
    def shuf(self , loader : list[T] , stage : Literal['init' , 'epoch'] = 'init') -> list[T]:
        '''shuffle at init or each epoch'''
        new_loader : Any = loader
        if stage == self.shufopt: 
            indices = np.random.permutation(np.arange(len(loader)))
            new_loader = [loader[i] for i in indices]
        return new_loader
    
class Checkpoint(MemFileStorage):
    '''model checkpoint for epochs'''
    def __init__(self, mem_storage: bool):
        super().__init__(mem_storage)
        self.epoch_queue : list[list] = []
        self.join_record : list[str]  = [] 
        # self.model_module = model_module

    def real_path(self , key : str): 
        return PATH.checkpoint.joinpath(f'{key}.pt')

    def epoch_key(self , epoch : int) -> str: 
        return f'ckpt.{self.model_key}.{epoch}'

    def new_model(self , model_param : dict , model_date : int):
        path = Path(model_param.get('path' , ''))
        self.model_key = f'{path.name}.{model_date}'
        self.epoch_queue = []
        self.join_record = [] 
        self.del_all()
    
    def join(self , src : Any , epoch : int , net : Any):
        if epoch < 0: 
            return
        if epoch >= len(self.epoch_queue): 
            self._extend_reliance()
        record_str = f'JOIN: Epoch {epoch}, from {src.__class__.__name__}({id(src)})'
        if src in self.epoch_queue[epoch]: 
            record_str += ', already exists'
        else:
            self.epoch_queue[epoch].append(src)
            record_str += ', append list'
        if self.exists(self.epoch_key(epoch)): 
            record_str += ', no need to save'
        else:
            self.save_state_dict(net , self.epoch_key(epoch))
            record_str += ', state dict saved'
        self.join_record.append(record_str)

    def disjoin(self , src : Any , epoch : int):
        if epoch < 0: 
            return
        record_str = f'DISJOIN: Epoch {epoch}, from {src.__class__.__name__}({id(src)})'
        
        if epoch >= len(self.epoch_queue):
            [Logger.stdout(record_str) for record_str in self.join_record]
            
        self.epoch_queue[epoch] = [s for s in self.epoch_queue[epoch] if s is not src]
        
        if self.epoch_queue[epoch]:
            record_str += f', {len(self.epoch_queue[epoch])} reliance left'
        else:
            self.del_one(self.epoch_key(epoch))
            record_str += f', delete state dict '
        self.join_record.append(record_str)

    def load_epoch(self , epoch : int) -> Any:
        assert epoch >= 0 , epoch
        if self.epoch_queue[epoch]:
            return self.load(self.epoch_key(epoch))
        else:
            [Logger.stdout(record_str) for record_str in self.join_record]
            raise Exception(f'no checkpoint of epoch {epoch}')
    
    def _extend_reliance(self , n_epochs = 200):
        self.epoch_queue += [[] for _ in range(n_epochs)] # extend epoch list

class Deposition:
    '''model saver'''
    def __init__(self , base_path : ModelPath):
        self.base_path = base_path

    def stack_model(self , model_dict : ModelDict , model_num , model_date , submodel = 'best'):
        model_dict.save(self.model_path(model_num , model_date , submodel) , stack = True)

    def dump_stacked_model(self , model_num , model_date , submodel = 'best'):
        model_path = self.model_path(model_num , model_date , submodel)
        for path in model_path.iterdir():
            if path.stem.endswith('.stack'):
                new_path = path.with_stem(path.stem.replace('.stack',''))
                new_path.unlink(missing_ok=True)
                path.rename(new_path)

    def load_model(self , model_num , model_date , submodel = 'best'):
        return self.base_path.model_file(model_num , model_date , submodel)
    
    def exists(self , model_num , model_date , submodel = 'best'):
        return self.base_path.exists(model_num , model_date , submodel)
    
    def model_path(self , model_num , model_date , submodel = 'best'):
        '''get model path of deposition giving model date / num / submodel'''
        return self.base_path.full_path(model_num , model_date , submodel)