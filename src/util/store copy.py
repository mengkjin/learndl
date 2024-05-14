import gc , itertools , os , torch
import numpy as np
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Any , Literal , Optional

from .config import TrainConfig
from ..func.basic import Filtered

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

    def load(self , path) -> Any:
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
        self.save(obj , path , group)
    
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

class Deposition:
    '''model saver'''
    def __init__(self , base_path : str):
        self.base_path = base_path

    def save_model(self , model_date , model_num , model_type = 'best' , state_dict : Optional[dict[str,torch.Tensor]] = None , **kwargs):
        ModelDict(self.model_path(model_date , model_num , model_type)).save(state_dict , **kwargs)

    def load_model(self , model_date , model_num , model_type = 'best'):
        return ModelDict(self.model_path(model_date , model_num , model_type))
    
    def exists(self , model_date , model_num , model_type = 'best'):
        return ModelDict(self.model_path(model_date , model_num , model_type)).exists()
    
    def model_path(self , model_date , model_num , model_type = 'best'):
        '''get model path of deposition giving model date / num / type'''
        return f'{self.base_path}/{model_num}/{model_date}/{model_type}'
    
    def model_iter(self , model_date_list , model_num_list , stage , resume = False):
        '''iter of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(model_date_list , model_num_list))
        if resume and stage == 'fit':
            models_trained = np.full(len(new_iter) , True , dtype = bool)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not self.exists(model_date , model_num):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = Filtered(new_iter , ~models_trained)
        return new_iter
    
    def model_dates(self , model_num , model_type = 'best'):
        '''get existing model dates'''
        path = f'{self.base_path}/{model_num}'
        return np.sort([int(p) for p in os.listdir(path) if os.path.exists(f'{path}/{p}/{model_type}/state_dict.pt')])

@dataclass
class ModelDict:
    model_path : str

    def __get(self , key : str) -> Any:
        path = f'{self.model_path}/{key}.pt'
        return torch.load(path , map_location='cpu') if os.path.exists(path) else None
        
    def exists(self) -> bool: return os.path.exists(f'{self.model_path}/state_dict.pt')
    def state_dict(self) -> Optional[dict[str,torch.Tensor]]: return self.__get('state_dict')
    def lgbt_model_string(self) -> Optional[str]: return self.__get('lgbt_model_string')

    def save(self , state_dict : Optional[dict[str,torch.Tensor]] , **kwargs):
        os.makedirs(self.model_path , exist_ok=True)
        if state_dict is not None:
            torch.save({key: value.cpu() for key, value in state_dict.items()} , f'{self.model_path}/state_dict.pt')
        for k , v in kwargs.items():
            assert k in dir(self) and v is not None , k
            torch.save(v , f'{self.model_path}/{k}.pt')