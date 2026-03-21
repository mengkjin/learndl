import torch
from src.proj import Logger , Proj
import numpy as np
from typing import Any

class MemoryManager():
    unit = 1024**3
    _instance = None

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , device = None , vb_level : Any = 2 , *args , **kwargs) -> None:
        self.initiate(device , vb_level , *args , **kwargs)

    @property
    def initiated(self) -> bool:
        return hasattr(self , 'cuda_avail')

    def initiate(self , device : torch.device | None = None , vb_level : Any = 2) -> None:
        if self.initiated:
            return
        self.vb_level = Proj.vb.level(vb_level)
        if device is not None:
            self.device = device
            self.gmem_total = self.gmem_total = torch.cuda.mem_get_info(self.device)[1] / self.unit if self.device.type == 'cuda' else 0.
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda' if device is None else device)
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        if self.device.type == 'cuda':
            self.gmem_total = torch.cuda.mem_get_info(self.device)[1] / self.unit
        elif self.device.type == 'mps':
            self.gmem_total = torch.mps.current_allocated_memory() / self.unit
        else:
            self.gmem_total = 0.
        self.record : dict[str, list] = {}

    def check(self , key = None, showoff = False , critical_ratio = 0.5):
        if self.device.type != 'cuda': 
            return 0.

        gmem_free = torch.cuda.mem_get_info(self.device)[0] / self.unit
        torch.cuda.empty_cache()
        if gmem_free > critical_ratio * self.gmem_total and not showoff: 
            # if showoff: 
            #     Logger.stdout(f'**Cuda Memory: Free {gmem_free:.1f}G' , vb_level = self.vb_level) 
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
            Logger.info(f'Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!' , vb_level = self.vb_level) 
        
        return gmem_free

    def show_memories(self , object_dict : dict[str,Any]):
        if not self.device.type == 'cuda': 
            return
        for key, value in object_dict.items():
            Logger.stdout(f'Cuda Memories of "{key}"     take {MemoryManager.object_memory(value):.4f}G' , indent = self.vb_level)
    
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
        elif isinstance(obj , dict):
            return sum([cls.object_memory(o , cuda_only = cuda_only) for o in obj.values()])
        elif isinstance(obj , object) and hasattr(obj , '__dict__'):
            return cls.object_memory(obj.__dict__ , cuda_only = cuda_only)
        else:
            return 0.
    
    @classmethod
    def tensor_memory(cls , tensor : torch.Tensor , cuda_only = True):
        if cuda_only and not tensor.is_cuda: 
            return 0.
        total_memory = tensor.element_size() * tensor.numel()
        return total_memory / cls.unit
    
    def print_memeory_record(self):
        if self.device.type == 'cuda' and self.record:
            info_dict = {k:f'{len(value)} counts, on average freed {np.mean(value):.2f}G' for k,value in self.record.items()}
            Logger.stdout_pairs(info_dict , title = 'Avg Freed Cuda Memory:')
                
    def clear_and_check(self , silent = True):
        gmem_free = torch.cuda.mem_get_info()[0] / self.unit
        torch.cuda.empty_cache()
        if not silent: 
            gmem_freed = torch.cuda.mem_get_info()[0] / self.unit - gmem_free
            gmem_free += gmem_freed
            gmem_allo  = torch.cuda.memory_allocated() / self.unit
            gmem_rsrv  = torch.cuda.memory_reserved() / self.unit
            Logger.success(f'Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!' , vb_level = self.vb_level) 
