import math
import numpy as np
import torch

from dataclasses import dataclass
from typing import Any
from .logger import *

use_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Use device name: ' + torch.cuda.get_device_name(0))

@dataclass
class BatchData:
    x       : torch.Tensor | tuple | list
    y       : torch.Tensor 
    w       : torch.Tensor | None
    i       : torch.Tensor 
    nonnan  : torch.Tensor 
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1:
            self.x = self.x[0]
        
    def to(self , device = None):
        return self.__class__(
            self.x.to(device) if isinstance(self.x , torch.Tensor) else type(self.x)(x.to(device) for x in self.x) , 
            self.y.to(device) , 
            None if self.w is None else self.w.to(device) , 
            self.i.to(device) , 
            self.nonnan.to(device) 
        )

    def cpu(self):
        return self.__class__(
            self.x.cpu() if isinstance(self.x , torch.Tensor) else type(self.x)(x.cpu() for x in self.x) , 
            self.y.cpu() , 
            None if self.w is None else self.w.cpu() , 
            self.i.cpu() , 
            self.nonnan.cpu() 
        )

    def cuda(self):
        return self.__class__(
            self.x.cuda() if isinstance(self.x , torch.Tensor) else type(self.x)(x.cuda() for x in self.x) , 
            self.y.cuda() , 
            None if self.w is None else self.w.cuda() , 
            self.i.cuda() , 
            self.nonnan.cuda() 
        )
    
@dataclass
class ModelOutputs:
    outputs : torch.Tensor | tuple | list

    def pred(self):
        if isinstance(self.outputs , (list , tuple)):
            return self.outputs[0]
        else:
            return self.outputs
    
    def hidden(self):
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None
        
class Device:
    torch_obj = (torch.Tensor , torch.nn.Module , torch.nn.ModuleList , torch.nn.ModuleDict , BatchData)

    def __init__(self , device = None) -> None:
        if device is None: device = use_device
        self.device = device
    def __call__(self, obj) -> Any:
        return self.send_to(obj , self.device)
    
    @classmethod
    def send_to(cls , x , device = None):
        if isinstance(x , (list,tuple)):
            return type(x)(cls.send_to(v , device) for v in x)
        elif isinstance(x , (dict)):
            return {k:cls.send_to(v , device) for k,v in x.items()}
        elif isinstance(x , cls.torch_obj): # maybe modulelist ... should be included
            return x.to(device)
        else:
            return x
        
    @classmethod
    def cpu(cls , x):
        if isinstance(x , (list,tuple)):
            return type(x)(cls.cpu(v) for v in x)
        elif isinstance(x , (dict)):
            return {k:cls.cpu(v) for k,v in x.items()}
        elif isinstance(x , cls.torch_obj): # maybe modulelist ... should be included
            return x.cpu()
        else:
            return x
    @classmethod
    def cuda(cls , x):
        if isinstance(x , (list,tuple)):
            return type(x)(cls.cuda(v) for v in x)
        elif isinstance(x , (dict)):
            return {k:cls.cuda(v) for k,v in x.items()}
        elif isinstance(x , cls.torch_obj): # maybe modulelist ... should be included
            return x.cuda()
        else:
            return x
        

    def torch_nans(self,*args,**kwargs):
        return torch.ones(*args , device = self.device , **kwargs).fill_(torch.nan)
    def torch_zeros(self,*args , **kwargs):
        return torch.zeros(*args , device = self.device , **kwargs)
    def torch_ones(self,*args,**kwargs):
        return torch.ones(*args , device = self.device , **kwargs)
    def torch_arange(self,*args,**kwargs):
        return torch.arange(*args , device = self.device , **kwargs)
    def print_cuda_memory(self):
        print(f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
              f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G')


class CosineScheduler:
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
                
class CustomBatchSampler(torch.utils.data.Sampler):
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