from __future__ import annotations

import gc , torch , shutil
import numpy as np
import pandas as pd

from dataclasses import dataclass , field
from pathlib import Path
from concurrent.futures import Future
from typing import Any , Literal , TypeVar

from src.proj import PATH
from src.proj.util import torch_load , Device , AsyncSaver
from .func import epoch_key
from .model_file import ModelDict
from .model_path import ModelPath

T = TypeVar('T')
class BufferStorage:
    '''Interface of mem or disk storage, methods'''
    def __init__(self , mem_storage : bool = False):
        self.is_mem = mem_storage
        self.memdisk : dict[str,Any] = {}
        self.records = pd.DataFrame(columns = pd.Index(['key' , 'path' , 'group']) , dtype = str)

    @property
    def keys(self): 
        return list(self.memdisk.keys())

    @property
    def to_disk(self):
        return self.is_mem

    def real_path(self , key : str): 
        return PATH.batch.joinpath(f'{key}.pt')
    
    def exists(self , key : str):
        if self.is_mem:
            return key in self.memdisk.keys()
        else:
            return self.real_path(key).exists()

    def save(self , obj : Any , key : str , group = 'default' , to_disk : bool | None = None):
        if self.is_mem:
            path = key
            self.memdisk[key] = Device.send_to(obj , 'cpu', copy = True)
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
    def __init__(self, loader_storage : BufferStorage , keys : list[str] , shuffle_option : Literal['static' , 'init' , 'epoch'] = 'static'):
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

@dataclass
class CkptEpochRecord:
    key : str
    path : Path
    state_dict : dict | None = None
    srcs : list = field(default_factory=list)

    def __bool__(self):
        return len(self.srcs) > 0

    def save(self):
        assert self.state_dict is not None , 'state_dict should not be None when saving'
        self._future = AsyncSaver.torch(self.state_dict , self.path , copy_for_safety = False )
        return self

    @property
    def future(self) -> Future:
        if not hasattr(self , '_future'):
            self.save()
        return self._future

    def add_src(self , src : Any):
        if src in self.srcs:
            return 'already exists'
        else:
            self.srcs.append(src)
            return 'added'

    def remove_src(self , src : Any):
        if src in self.srcs:
            self.srcs = list(set(self.srcs) - {src})
            if len(self.srcs) == 0:
                self.state_dict = None
            return f'removed and {len(self.srcs)} reliance left'
        else:
            return 'not found'

    def clear(self):
        self.state_dict = None
        if hasattr(self , '_future'):
            self._future.cancel()
        self.path.unlink(missing_ok=True)
        self.srcs.clear()
        return self

class Checkpoint:
    '''model checkpoint for epochs'''
    def __init__(self , * , memory_buffer_epochs = 20):
        """
        mem_storage: whether to store in memory
        num_epochs: minimum number of epochs to store
        """
        self.memory_buffer_epochs = memory_buffer_epochs
        self.epoch_maps : dict[str , CkptEpochRecord] = {}
        self.join_messages : list[str] = [] 

    def exists(self , epoch : int , phase : int = 0):
        ep_key = epoch_key(epoch , phase)
        if ep_key not in self.epoch_maps:
            return False
        if self.epoch_maps[ep_key].state_dict is not None:
            return True
        if self.epoch_maps[ep_key].future.done() and self.epoch_maps[ep_key].path.exists():
            return True
        return False

    def buffer(self , epoch : int , phase : int = 0 , state_dict : dict | None = None):
        if state_dict is None and epoch not in self.epoch_maps:
            raise Exception(f'state_dict should not be None when epoch {epoch} is not in epoch_maps')
        state_dict = Device.send_to(state_dict, 'cpu', copy = True)
        ep_key = epoch_key(epoch , phase)
        key = f'{self.model_key}.{ep_key}'
        path = PATH.checkpoint.joinpath(f'{key}.pt')
        self.epoch_maps[ep_key] = CkptEpochRecord(key , path , state_dict).save()

    def load(self , epoch : int , phase : int = 0) -> Any:
        if epoch < 0 or phase < 0:
            return {}
        ep_key = epoch_key(epoch , phase)
        record = self.epoch_maps[ep_key]
        if record.state_dict is not None:
            return record.state_dict
        else:
            _ = record.future.result()
            return torch_load(record.path)

    def clear_all(self):
        self.join_messages.clear()
        for record in self.epoch_maps.values():
            record.clear()
        for path in PATH.checkpoint.iterdir():
            path.unlink(missing_ok=True)
        self.epoch_maps.clear()
        gc.collect()

    def new_model(self , model_num : int , model_date : int , next_attempt : int , next_redo : int , **kwargs):
        self.model_key = f'ckpt.{model_num}.{model_date}.trial{next_attempt}-{next_redo}'
        self.clear_all()

    def auto_save(self , state_dict : dict):
        epoch = state_dict['epoch']
        phase = state_dict.get('phase', 0)
        self.join(self , epoch , phase , state_dict)
        self.disjoin(self , epoch , phase - 1)
        self.disjoin(self , epoch - self.memory_buffer_epochs , phase)

    def join(self , src : Any , epoch : int , phase : int , state_dict : Any | None = None):
        if epoch < 0: 
            return
        ep_key = epoch_key(epoch , phase)
        if ep_key not in self.epoch_maps:
            self.buffer(epoch , phase , state_dict)
        record = self.epoch_maps[ep_key]

        record_str = f'JOIN: {ep_key}, from {src.__class__.__name__}({id(src)}), {record.add_src(src)}'
        self.join_messages.append(record_str)

    def disjoin(self , src : Any , epoch : int , phase : int):
        if epoch < 0 or phase < 0: 
            return
        ep_key = epoch_key(epoch , phase)
        if ep_key not in self.epoch_maps:
            return
            
        record = self.epoch_maps[ep_key]
        record_str = f'DISJOIN: {ep_key}, from {src.__class__.__name__}({id(src)}), {record.remove_src(src)}'
        self.join_messages.append(record_str)
    
class Deposition:
    '''model saver'''
    def __init__(self , base_path : ModelPath):
        self.base_path = base_path

    def shrink_key(self , key : str) -> str:
        return key.replace('.' , '-').replace(' ' , '')

    def stack_model(self , model_dict : ModelDict , attempt_key : str , model_num , model_date , submodel = 'best'):
        assert attempt_key , 'attempt_key is required when stacking model'
        attempt_key = self.shrink_key(attempt_key)
        model_path = self.model_path(model_num , model_date , submodel)
        if attempt_key:
            model_path = model_path.joinpath(attempt_key.replace('.' , '-').replace(' ' , ''))
        model_dict.save(model_path)

    def dump_stacked_model(self , attempt_key : str , model_num , model_date , submodel = 'best'):
        model_path = self.model_path(model_num , model_date , submodel)
        attempt_key = self.shrink_key(attempt_key)
        attempt_path = model_path.joinpath(attempt_key)
        assert attempt_path.exists() , f'attempt_path {attempt_path} does not exist'
        for path in attempt_path.iterdir():
            new_path = model_path.joinpath(path.name)
            new_path.unlink(missing_ok=True)
            path.rename(new_path)
        self.clear_stacked_models(model_num , model_date , submodel)

    def clear_stacked_models(self , model_num , model_date , submodel = 'best'):
        model_path = self.model_path(model_num , model_date , submodel)
        for path in model_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)

    def load_model(self , model_num , model_date , submodel = 'best'):
        return self.base_path.model_file(model_num , model_date , submodel)
    
    def exists(self , model_num , model_date , submodel = 'best'):
        return self.base_path.exists(model_num , model_date , submodel)
    
    def model_path(self , model_num , model_date , submodel = 'best'):
        '''get model path of deposition giving model date / num / submodel'''
        return self.base_path.full_path(model_num , model_date , submodel)