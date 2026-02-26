import torch
from datetime import datetime
from src.proj import Logger
import numpy as np

class MemoryManager():
    unit = 1024**3

    def __init__(self , device = 0) -> None:
        self.cuda_avail = torch.cuda.is_available()
        if self.cuda_avail:
            self.device = torch.device(device)
            self.unit = type(self).unit
            if self.cuda_avail: 
                self.gmem_total = torch.cuda.mem_get_info(self.device)[1] / self.unit
            self.record = {}

    def check(self , key = None, showoff = False , critical_ratio = 0.5 , starter = '**'):
        if not self.cuda_avail: 
            return 0.

        gmem_free = torch.cuda.mem_get_info(self.device)[0] / self.unit
        torch.cuda.empty_cache()
        if gmem_free > critical_ratio * self.gmem_total and not showoff: 
            # if showoff: 
            #     Logger.stdout(f'**Cuda Memory: Free {gmem_free:.1f}G') 
            return gmem_free
        
        gmem_freed = torch.cuda.mem_get_info(self.device)[0] / self.unit - gmem_free
        gmem_free += gmem_freed
        gmem_allo  = torch.cuda.memory_allocated(self.device) / self.unit
        gmem_rsrv  = torch.cuda.memory_reserved(self.device) / self.unit
        
        if key is not None:
            if key not in self.record.keys(): 
                self.record[key] = []
            self.record[key].append(gmem_freed)
        if showoff: 
            Logger.stdout(f'{starter}{datetime.now()}Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!') 
        
        return gmem_free
    
    def collect(self):
        torch.cuda.empty_cache() # collect graphic memory 
        # gc.collect() # collect memory, very slow
    
    def __bool__(self):
        return True
    
    @classmethod
    def object_memory(cls , obj , cuda_only = True):
        if isinstance(obj , torch.Tensor):
            return cls.tensor_memory(obj , cuda_only = cuda_only)
        elif isinstance(obj , (list,tuple)):
            return sum([cls.object_memory(o , cuda_only = cuda_only) for o in obj])
        elif isinstance(obj , object):
            return cls.object_memory(obj.__dict__ , cuda_only = cuda_only)
        elif isinstance(obj , dict):
            return sum([cls.object_memory(o , cuda_only = cuda_only) for o in obj.values()])
        else:
            return 0.
    
    @classmethod
    def tensor_memory(cls , tensor , cuda_only = True):
        if cuda_only and not tensor.is_cuda: 
            return 0.
        total_memory = tensor.element_size() * tensor.numel()
        return total_memory / cls.unit
    
    def print_memeory_record(self):
        if self.cuda_avail:
            Logger.stdout(f' Avg Freed Cuda Memory: ')
            for key , value in self.record.items():
                Logger.stdout(f'{key} : {len(value)} counts, on average freed {np.mean(value):.2f}G' , indent = 1)
    
    @classmethod
    def clear_and_check(cls , silent = True):
        gmem_free = torch.cuda.mem_get_info()[0] / cls.unit
        torch.cuda.empty_cache()
        if not silent: 
            gmem_freed = torch.cuda.mem_get_info()[0] / cls.unit - gmem_free
            gmem_free += gmem_freed
            gmem_allo  = torch.cuda.memory_allocated() / cls.unit
            gmem_rsrv  = torch.cuda.memory_reserved() / cls.unit
            Logger.stdout(f'Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!') 

    @staticmethod
    def except_MemoryError(func , out = None , print_str = ''):
        def wrapper(*args , **kwargs):
            try:
                value = func(*args , **kwargs)
            except torch.cuda.OutOfMemoryError:
                Logger.warning(f'OutOfMemoryError on {print_str}')
                torch.cuda.empty_cache()
                value = out
            return value
        return wrapper