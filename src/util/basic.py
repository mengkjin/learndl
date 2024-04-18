import torch
import numpy as np
import pandas as pd
import gc , time , os , psutil

from copy import deepcopy
from torch import Tensor
from typing import Any , Literal , Optional

use_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Use device name: ' + torch.cuda.get_device_name(0))
        
class Device:
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
        elif hasattr(x , 'to'): # maybe modulelist ... should be included
            return x.to(device)
        else:
            return x
        
    @classmethod
    def cpu(cls , x):
        if isinstance(x , (list,tuple)):
            return type(x)(cls.cpu(v) for v in x)
        elif isinstance(x , (dict)):
            return {k:cls.cpu(v) for k,v in x.items()}
        elif hasattr(x , 'cpu'): # maybe modulelist ... should be included
            return x.cpu()
        else:
            return x
    @classmethod
    def cuda(cls , x):
        if isinstance(x , (list,tuple)):
            return type(x)(cls.cuda(v) for v in x)
        elif isinstance(x , (dict)):
            return {k:cls.cuda(v) for k,v in x.items()}
        elif hasattr(x , 'cuda'): # maybe modulelist ... should be included
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

class Timer:
    def __init__(self , *args):
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        print(self.key , '...', end='')
    def __exit__(self, type, value, trace):
        print(f'... cost {time.time()-self.start_time:.2f} secs')
            
class MemoryPrinter:
    def __init__(self) -> None:
        pass
    def __repr__(self) -> str:
        return 'Used: {:.2f}G; Free {:.2f}G'.format(
            float(psutil.virtual_memory().used)/1024**3,
            float(psutil.virtual_memory().free)/1024**3)
    def print(self):
        print(self.__repr__())
        
        