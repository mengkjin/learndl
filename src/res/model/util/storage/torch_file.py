"""
Torch file storage for trainer
"""
from __future__ import annotations

import gc , torch
import numpy as np
import pandas as pd

from typing import Any , Literal , TypeVar , TypeAlias 
from collections.abc import Sequence

from src.proj import PATH , Load
from src.proj.util.functional.device import Device

__all__ = ['TorchFileStorage' , 'StoredTorchFileLoader']

T = TypeVar('T')
ShuffleTime : TypeAlias = Literal['init' , 'epoch']
ShuffleOption : TypeAlias = Literal['static' , 'init' , 'epoch']

class TorchFileStorage:
    """Interface of mem or disk storage, methods"""
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
        return PATH.minibatch.joinpath(f'{key}.pt')
    
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
            return Load.torch(path) if path.exists() else None
    
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

class StoredTorchFileLoader(Sequence):
    """'retrieve batch_input from a Storage"""
    def __init__(self, loader_storage : TorchFileStorage , keys : list[str] , shuffle_option : ShuffleOption = 'static'):
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
    def shuf(self , loader : list[T] , stage : ShuffleTime = 'init') -> list[T]:
        """shuffle at init or each epoch"""
        new_loader : Any = loader
        if stage == self.shufopt: 
            indices = np.random.permutation(np.arange(len(loader)))
            new_loader = [loader[i] for i in indices]
        return new_loader