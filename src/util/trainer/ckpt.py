import os

from torch import nn
from typing import Any , Literal
from ..store import Storage

class Checkpoint(Storage):
    n_epochs = 200

    def __init__(self, store_type: Literal['mem' , 'disk']):
        super().__init__(store_type)
        self.reliance = [[] for _ in range(self.n_epochs)]
        self.join_record = []

    def new_model(self , model_param : dict , model_date : int):
        if self.is_disk:
            self.dir = '{}/{}'.format(model_param.get('path') , model_date)
        else:
            self.dir = '{}/{}'.format(os.path.basename(str(model_param.get('path'))) , model_date)
        self.reliance = [[] for _ in range(self.n_epochs)]
        self.join_record = []
        self.del_all()
    
    def join(self , src : Any , epoch : int , net : nn.Module):
        if epoch < 0: return
        if epoch >= len(self.reliance):
            self.reliance += [[] for _ in range(self.n_epochs)] # extend epoch list
        record_str = f'JOIN: Epoch {epoch}, from {src.__class__}({id(src)})'
        if src not in self.reliance[epoch]: 
            self.reliance[epoch].append(src)
            record_str += ', append list'
        else:
            record_str += ', already exists'
        if not self.exists(self.epoch_path(epoch)): 
            self.save_state_dict(net , self.epoch_path(epoch))
            record_str += ', state dict saved'
        else:
            record_str += ', no need to save'
        self.join_record.append(record_str)

    def disjoin(self , src , epoch : int):
        if epoch < 0: return
        record_str = f'DISJOIN: Epoch {epoch}, from {src.__class__}({id(src)})'
        self.reliance[epoch] = [s for s in self.reliance[epoch] if s is not src]
        if len(self.reliance[epoch]) == 0:  
            self.del_path(self.epoch_path(epoch))
            record_str += f', delete state dict '
        else:
            record_str += f', {len(self.reliance[epoch])} reliance left'
        self.join_record.append(record_str)

    def load_epoch(self , epoch):
        if len(self.reliance[epoch]) == 0:
            for i , rel in enumerate(self.reliance):
                if len(rel) > 0: print(i , rel)
            [print(record_str) for record_str in self.join_record]
            raise Exception(f'no connection of epoch {epoch}')
        return self.load(self.epoch_path(epoch))
    
    def epoch_path(self , epoch):
        return f'{self.dir}/checkpoint.{epoch}.pt'