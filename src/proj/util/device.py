"""Torch device selection, nested ``.to()``, and simple RAM usage string."""
from __future__ import annotations
import psutil , torch
from torch.nn import Module
from typing import Any

from src.proj.log import Logger

__all__ = ['Device' , 'MemoryPrinter']

# MPS memory management
if torch.backends.mps.is_available() == 'mps':
    # clear MPS cache
    torch.mps.empty_cache()
    
    # set memory allocation strategy
    torch.mps.set_per_process_memory_fraction(0.7)  # 使用60%的GPU内存

def send_to(x : Any , device = None) -> Any:
    """Recursively move tensors/modules/containers to ``device``."""
    if isinstance(x , torch.Tensor | Module):
        return x.to(device)
    if isinstance(x , (list,tuple)):
        return type(x)(send_to(v , device) for v in x)
    elif isinstance(x , (dict)):
        return {k:send_to(v , device) for k,v in x.items()}
    elif hasattr(x , 'to'): # maybe modulelist ... self defined class
        return x.to(device)
    else:
        return x
    
def get_device(obj : Module | torch.Tensor | list | tuple | dict | Any) -> torch.device:
    """Infer ``torch.device`` from tensor, module, or nested structure."""
    if isinstance(obj , torch.Tensor):
        return obj.device
    elif isinstance(obj , Module):
        return next(obj.parameters()).device
    elif isinstance(obj , (list,tuple)):
        return get_device(obj[0])
    elif isinstance(obj , dict):
        return get_device(list(obj.values()))
    elif hasattr(obj , 'device'):
        return obj.device
    else:
        raise ValueError(f'{obj} is not a valid object')

class Device:
    """Preferred accelerator (MPS > CUDA > CPU) with helpers to move data."""

    def __init__(self , try_cuda = True , try_mps = True) -> None:
        """Pick device via ``use_device`` (MPS checked before CUDA when available)."""
        self.device = self.use_device(try_cuda)
    def __repr__(self): return str(self.device)
    def __call__(self, obj):
        """Move ``obj`` to this device (see ``send_to``)."""
        return send_to(obj , self.device)
    def __eq__(self , other : Any) -> bool:
        """True if type and index match another ``Device`` or ``torch.device``."""
        return self.compare_devices(self.device , other)

    @staticmethod
    def compare_devices(device1 : torch.device | Device , device2 : torch.device | Device) -> bool:
        dev1 = device1 if isinstance(device1 , torch.device) else device1.device
        dev2 = device2 if isinstance(device2 , torch.device) else device2.device
        return (dev1.type == dev2.type) and ((dev1.index or 0) == (dev2.index or 0))

    @classmethod
    def use_device(cls , try_cuda = True) -> torch.device:
        """Return ``mps``, ``cuda:0`` if allowed, else ``cpu``."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif try_cuda and torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')

    @property
    def is_cuda(self):
        """True if using CUDA."""
        return self.device.type == 'cuda'
    @property
    def is_mps(self):
        """True if using MPS."""
        return self.device.type == 'mps'
    @property
    def is_cpu(self):
        """True if using CPU."""
        return self.device.type == 'cpu'
        
    def cpu(self , x):
        """Move ``x`` to CPU."""
        return send_to(x , 'cpu')
    def cuda(self , x):
        """Move ``x`` to CUDA device 0."""
        return send_to(x , 'cuda')
    def mps(self , x):
        """Move ``x`` to MPS."""
        return send_to(x , 'mps')
    
    def status(self):
        """Log allocator stats for CUDA/MPS or a short CPU message."""
        if self.is_cuda:
            Logger.stdout(f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
                  f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G')
        elif self.is_mps:
            Logger.stdout(f'Allocated {torch.mps.current_allocated_memory() / 1024**3:.1f}G, '+\
                  f'Driver Allocated {torch.mps.driver_allocated_memory() / 1024**3:.1f}G')
        else:
            Logger.stdout(f'Not using cuda or mps {self.device}')

    @staticmethod
    def send_to(x : Any , device = None) -> Any:
        """Module-level ``send_to`` bound for convenience."""
        return send_to(x , device)

    @staticmethod
    def get_device(obj : Module | torch.Tensor | list | tuple | dict | Any) -> torch.device:
        """Module-level ``get_device`` bound for convenience."""
        return get_device(obj)
        
class MemoryPrinter:
    """Tiny repr helper for host RAM used/free (via psutil)."""

    def __repr__(self) -> str:
        return 'Used: {:.2f}G; Free {:.2f}G'.format(
            float(psutil.virtual_memory().used)/1024**3,
            float(psutil.virtual_memory().free)/1024**3)
        
        