import torch
import numpy as np
import gc , math , time
from copy import deepcopy
from ..functional.func import *

class timer:
    def __init__(self , *args):
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        print(self.key , '...', end='')
    def __exit__(self, type, value, trace):
        print(f'... cost {time.time()-self.start_time:.2f} secs')
        
class FilteredIterator:
    def __init__(self, iterable, condition):
        self.iterable = iterable
        self.condition = condition

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item

class lr_cosine_scheduler:
    def __init__(self , optimizer , warmup_stage = 10 , anneal_stage = 40 , initial_lr_div = 10 , final_lr_div = 1e4):
        self.warmup_stage= warmup_stage
        self.anneal_stage= anneal_stage
        self.optimizer = optimizer
        self.base_lrs = [x['lr'] for x in optimizer.param_groups]
        self.initial_lr= [x / initial_lr_div for x in self.base_lrs]
        self.final_lr= [x / final_lr_div for x in self.base_lrs]
        self.last_epoch = 0
        self._step_count= 1
        self._linear_phase = self._step_count / self.warmup_stage
        self._cos_phase = math.pi / 2 * (self._step_count - self.warmup_stage) / self.anneal_stage
        self._last_lr= self.initial_lr
        
    def get_last_lr(self):
        #Return last computed learning rate by current scheduler.
        return self._last_lr

    def state_dict(self):
        #Returns the state of the scheduler as a dict.
        return self.__dict__
    
    def step(self):
        self.last_epoch += 1
        if self._step_count <= self.warmup_stage:
            self._last_lr = [y+(x-y)*self._linear_phase for x,y in zip(self.base_lrs,self.initial_lr)]
        elif self._step_count <= self.warmup_stage + self.anneal_stage:
            self._last_lr = [y+(x-y)*math.cos(self._cos_phase) for x,y in zip(self.base_lrs,self.final_lr)]
        else:
            self._last_lr = self.final_lr
        for x , param_group in zip(self._last_lr,self.optimizer.param_groups):
            param_group['lr'] = x
        self._step_count += 1
        self._linear_phase = self._step_count / self.warmup_stage
        self._cos_phase = math.pi / 2 * (self._step_count - self.warmup_stage) / self.anneal_stage
                
class DesireBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler , batch_size_list , drop_res = True):
        self.sampler = sampler
        self.batch_size_list = np.array(batch_size_list).astype(int)
        assert (self.batch_size_list >= 0).all()
        self.drop_res = drop_res
        
    def __iter__(self):
        if (not self.drop_res) and (sum(self.batch_size_list) < len(self.sampler)):
            new_list = np.append(self.batch_size_list , len(self.sampler) - sum(self.batch_size_list))
        else:
            new_list = self.batch_size_list
        
        batch_count , sample_idx = 0 , 0
        while batch_count < len(new_list):
            if new_list[batch_count] > 0:
                batch = [0] * new_list[batch_count]
                idx_in_batch = 0
                while True:
                    batch[idx_in_batch] = self.sampler[sample_idx]
                    idx_in_batch += 1
                    sample_idx +=1
                    if idx_in_batch == new_list[batch_count]:
                        yield batch
                        break
            batch_count += 1
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self):
        if self.batch_size_list.sum() < len(self.sampler):
            return len(self.batch_size_list) + 1 - self.drop_res
        else:
            return np.where(self.batch_size_list.cumsum() >= len(self.sampler))[0][0] + 1
        
class versatile_storage():
    def __init__(self , *args):
        if len(args) > 0 and args[0] in ['disk' , 'mem']:
            self.activate(args[0])

    def activate(self , default = 'mem'):
        assert default in ['disk' , 'mem']
        self.default = default
        self.mem_disk = dict()
        self.file_record = list()
        self.file_group = dict()
    
    def save(self , obj , paths , to_disk = False , group = 'default'):
        for p in self._pathlist(paths): 
            self._saveone(obj , p , self.default == 'disk' or to_disk)
            self._addrecord(p , group)
            
    def load(self , path , from_disk = False):
        return torch.load(path) if self.default == 'disk' or from_disk else self.mem_disk[path]

    def _pathlist(self , p):
        if p is None: return []
        return [p] if isinstance(p , str) else p
    
    def _saveone(self , obj , p , to_disk = False):
        if to_disk:
            torch.save(obj , p)
        else:
            self.mem_disk[p] = deepcopy(obj)
    
    def _addrecord(self , p , group):
        self.file_record = np.union1d(self.file_record , p)
        if group not in self.file_group.keys(): 
            self.file_group[group] = [p]
        else:
            self.file_group[group] = np.union1d(self.file_group[group] , [p])
    
    def save_model_state(self , model , paths , to_disk = False , group = 'default'):
        sd = model.state_dict() if (self.default == 'disk' or to_disk) else deepcopy(model).cpu().state_dict()
        self.save(sd , paths , to_disk , group)
        
    def load_model_state(self , model , path , from_disk = False):
        sd = self.load(path , from_disk)
        model.load_state_dict(sd)
        return model
            
    def valid_paths(self , paths):
        return np.intersect1d(self._pathlist(paths) ,  self.file_record).tolist()
    
    def del_path(self , *args):
        for paths in args:
            if self.default == 'disk':
                [os.remove(p) for p in self._pathlist(paths) if os.path.exists(p)]
            else:
                [self.mem_disk.__delitem__(p) for p in np.intersect1d(self._pathlist(paths) , list(self.mem_disk.keys()))]
            self.file_record = np.setdiff1d(self.file_record , paths)
        gc.collect()
        
    def del_group(self , clear_groups = []):
        for g in self._pathlist(clear_groups):
            paths = self.file_group.get(g)
            if paths is not None:
                self.del_path(paths)
                del self.file_group[g]