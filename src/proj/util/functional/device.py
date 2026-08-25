"""Torch device selection, nested ``.to()``, inference policy, and CUDA OOM redirect."""
from __future__ import annotations

import torch
from copy import deepcopy
from typing import Any, Literal

from src.proj.core import lit
from src.proj.bases import BoundLogger

__all__ = ['Device', 'DevicePolicy']

DevicePolicy = Literal['cuda_then_cpu', 'cuda_then_raise', 'cpu_only']
InferenceStage = Literal['test', 'predict', 'retrospective']
_VALID_POLICIES: tuple[DevicePolicy, ...] = ('cuda_then_cpu', 'cuda_then_raise', 'cpu_only')
_VALID_STAGES: tuple[InferenceStage, ...] = ('test', 'predict', 'retrospective')

def _send_to(x : Any , device = None , copy = False) -> Any:
    """Recursively move tensors/modules/containers to ``device``."""
    if hasattr(x , 'to'): # maybe modulelist ... self defined class
        x_ = x.to(device)
        if copy and x_ is x:
            x_ = deepcopy(x_)
        return x_
    if isinstance(x , (list,tuple)):
        return type(x)(_send_to(v , device, copy) for v in x)
    elif isinstance(x , (dict)):
        return {k:_send_to(v , device, copy) for k,v in x.items()}
    else:
        return deepcopy(x) if copy else x
    
def _get_device(obj : torch.Module | torch.Tensor | list | tuple | dict | Any) -> torch.device:
    """Infer ``torch.device`` from tensor, module, or nested structure."""
    if isinstance(obj , torch.Tensor):
        return getattr(obj, 'device')
    elif isinstance(obj , torch.nn.Module):
        return next(getattr(obj, 'parameters')()).device
    elif isinstance(obj , (list,tuple)):
        return _get_device(obj[0])
    elif isinstance(obj , dict):
        return _get_device(list(obj.values()))
    elif hasattr(obj , 'device'):
        return getattr(obj, 'device')
    else:
        raise ValueError(f'{obj} is not a valid object')
    
class Device(BoundLogger):
    """Preferred accelerator (MPS > CUDA > CPU) with helpers to move data."""
    def __init__(
        self ,
        try_cuda = True ,
        policy : DevicePolicy = 'cuda_then_raise' ,
        * ,
        indent : int = 0 ,
        vb_level : lit.VerbosityLevel = 1 ,
        **kwargs,
    ):
        """Pick device via ``use_device`` (MPS checked before CUDA when available)."""
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.set_mps_memory_fraction()
        if policy not in _VALID_POLICIES:
            raise ValueError(f'Invalid device policy {policy}, expected one of {_VALID_POLICIES}')
        self.policy : DevicePolicy = policy
        self.try_cuda = False if policy == 'cpu_only' else try_cuda
        self._redirected = False
        self._device = self.use_device(self.try_cuda)

    def __repr__(self): 
        return str(self.device)

    def __call__(self, obj):
        """Move ``obj`` to this device, applying inference policy on CUDA OOM."""
        return self.to(obj)
    def __eq__(self , other : Any) -> bool:
        """True if type and index match another ``Device`` or ``torch.device``."""
        if isinstance(other , (torch.device, Device)):
            return self.compare_devices(self.device , other)
        return False

    @classmethod
    def set_mps_memory_fraction(cls , fraction : float = 0.7):
        if torch.backends.mps.is_available() == 'mps':
            # clear MPS cache
            torch.mps.empty_cache()
            
            # set memory allocation strategy
            torch.mps.set_per_process_memory_fraction(fraction)  # 使用60%的GPU内存

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def redirected(self) -> bool:
        """True after a CUDA OOM fallback to CPU."""
        return self._redirected

    @staticmethod
    def compare_devices(device1 : torch.device | Device , device2 : torch.device | Device) -> bool:
        dev1 = device1.device if isinstance(device1 , Device) else device1
        dev2 = device2.device if isinstance(device2 , Device) else device2
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

    @classmethod
    def from_inference_stage(cls , stage : str) -> Device:
        """Build a Device from ``configs/preference/gpu.yaml`` and current fit-lock occupancy."""
        from src.proj.env.machine import MACHINE
        from src.proj.util.script.fit_lock import FitLock

        if stage not in _VALID_STAGES:
            raise ValueError(f'Invalid inference stage {stage}, expected one of {_VALID_STAGES}')
        occupancy = 'busy' if FitLock.is_held() else 'idle'
        policy = MACHINE.preference(
            'gpu' , f'inference/{stage}/{occupancy}' , default = 'cuda_then_raise'
        )
        if policy not in _VALID_POLICIES:
            raise ValueError(f'Invalid device policy {policy} for {stage}/{occupancy}')
        return cls(try_cuda = policy != 'cpu_only' , policy = policy)

    @staticmethod
    def is_cuda_oom(exc : BaseException) -> bool:
        """True if ``exc`` is a CUDA out-of-memory error."""
        if isinstance(exc , torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc , RuntimeError) and 'out of memory' in str(exc).lower()

    def to(self , obj : Any) -> Any:
        """Move ``obj`` to the current device; ``cuda_then_cpu`` redirects on OOM."""
        try:
            return _send_to(obj , self.device)
        except Exception as exc:
            if self.policy != 'cuda_then_cpu' or not self.is_cuda_oom(exc):
                raise
            self.redirect_to_cpu()
            return _send_to(obj , self.device)

    def redirect_to_cpu(self) -> None:
        """Switch this Device to CPU after CUDA failure; subsequent transfers follow."""
        if self.is_cpu:
            self._redirected = True
            return
        prev = self._device
        self._device = torch.device('cpu')
        self.try_cuda = False
        self._redirected = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.warning(f'Device redirected {prev} -> cpu (policy={self.policy})')

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
        return _send_to(x , 'cpu')
    def cuda(self , x):
        """Move ``x`` to CUDA device 0."""
        return _send_to(x , 'cuda')
    def mps(self , x):
        """Move ``x`` to MPS."""
        return _send_to(x , 'mps')
    
    def status(self):
        """Log allocator stats for CUDA/MPS or a short CPU message."""
        if self.is_cuda:
            info = f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
                  f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G'
        elif self.is_mps:
            info = f'Allocated {torch.mps.current_allocated_memory() / 1024**3:.1f}G, '+\
                  f'Driver Allocated {torch.mps.driver_allocated_memory() / 1024**3:.1f}G'
        else:
            info = f'Not using cuda or mps {self.device}'
        self.logger.stdout(info)

    @staticmethod
    def send_to(x : Any , device = None , copy = False) -> Any:
        """Module-level ``send_to`` bound for convenience."""
        return _send_to(x , device)

    @staticmethod
    def get_device(obj : torch.Module | torch.Tensor | list | tuple | dict | Any) -> torch.device:
        """Module-level ``get_device`` bound for convenience."""
        return _get_device(obj)
