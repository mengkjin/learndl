import os

from torch import nn
from typing import Any , Literal
from ..store import Storage

class Checkpoint(Storage):
    ''''''
    def __init__(self, store_type: Literal['mem' , 'disk']):
        super().__init__(store_type)
        self.epoch_queue : list[list]   = []
        self.join_record : list[str] = [] 

    def new_model(self , model_param : dict , model_date : int):
        if self.is_disk:
            self.dir = '{}/{}'.format(model_param.get('path') , model_date)
        else:
            self.dir = '{}/{}'.format(os.path.basename(str(model_param.get('path'))) , model_date)
        self.epoch_queue = []
        self.join_record = [] 
        self.del_all()
    
    def join(self , src : Any , epoch : int , net : nn.Module):
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
        assert epoch > 0 , epoch
        if self.epoch_queue[epoch]:
            self.load(self.epoch_path(epoch))
        else:
            [print(record_str) for record_str in self.join_record]
            raise Exception(f'no checkpoint of epoch {epoch}')
    
    def epoch_path(self , epoch):
        return f'{self.dir}/checkpoint.{epoch}.pt'
    
    def _extend_reliance(self , n_epochs = 200):
        self.epoch_queue += [[] for _ in range(n_epochs)] # extend epoch list