import os

from torch import nn
from typing import Any
from ..store import Storage

class Checkpoints(Storage):
    n_epochs = 20

    def __init__(self, mem_storage: bool = True , ):
        super().__init__(mem_storage)

    def new_model(self , model_param : dict , model_date : int):
        if self.is_disk:
            self.dir = '{}/{}'.format(model_param.get('path') , model_date)
        else:
            self.dir = '{}/{}'.format(os.path.basename(str(model_param.get('path'))) , model_date)
        self.sources = [[] for _ in range(self.n_epochs)]
    
    def join(self , src : Any , epoch : int , net : nn.Module):
        if epoch > len(self.sources):
            self.sources += [[] for _ in range(self.n_epochs)]
        if len(self.sources[epoch]) == 0:  
            self.save_state_dict(net , self.epoch_path(epoch))
        if src not in self.sources[epoch]: 
            self.sources[epoch].append(src)

    def disjoin(self , src , epoch : int):
        if epoch is not None:
            self.sources[epoch] = [s for s in self.sources[epoch] if s is not src]
            if len(self.sources[epoch]) == 0:  self.del_path(self.epoch_path(epoch))

    def load_epoch(self , epoch):
        return self.load(self.epoch_path(epoch))
    
    def epoch_path(self , epoch):
        return f'{self.dir}/checkpoint.{epoch}.pt'