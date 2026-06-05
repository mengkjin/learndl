from __future__ import annotations

import gc

from concurrent.futures import Future
from dataclasses import dataclass , field
from pathlib import Path
from typing import Any

from src.proj import PATH
from src.proj.util.io.torch_load import torch_load
from src.proj.util.functional.device import Device
from src.proj.util.io.async_save import AsyncSaver
from src.res.model.util.core import epoch_key

@dataclass
class CkptEpochRecord:
    key : str
    path : Path
    state_dict : dict | None = None
    srcs : list = field(default_factory=list)

    def __post_init__(self):
        self.saved = False

    def __bool__(self):
        return len(self.srcs) > 0

    def save(self):
        if self.saved:
            return self
        self.future : Future = AsyncSaver.torch(self.state_dict , self.path , copy_for_safety = False)
        self.saved = True
        return self

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
        if hasattr(self , 'future'):
            self.future.cancel()
        self.path.unlink(missing_ok=True)
        self.srcs.clear()
        return self

class Checkpoint:
    """model checkpoint for epochs"""
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