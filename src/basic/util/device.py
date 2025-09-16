import psutil , torch
from torch.nn import Module
from typing import Any

if torch.backends.mps.is_available():
    use_device = torch.device('mps')
elif torch.cuda.is_available():
    use_device = torch.device('cuda:0')
else:
    use_device = torch.device('cpu')
        
# MPS memory management
if use_device.type == 'mps':
    # clear MPS cache
    torch.mps.empty_cache()
    
    # set memory allocation strategy
    torch.mps.set_per_process_memory_fraction(0.7)  # 使用60%的GPU内存

def send_to(x : Any , device = None) -> Any:
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
    
def get_device(obj : Module | torch.Tensor | list | tuple | dict | Any):
    if isinstance(obj , torch.Tensor):
        return obj.device
    elif isinstance(obj , Module):
        return next(obj.parameters()).device
    elif isinstance(obj , (list,tuple)):
        return get_device(obj[0])
    elif isinstance(obj , dict):
        return get_device(list(obj.values()))
    elif obj.__class__.__name__ == 'BatchData':
        return obj.device
    elif obj.__class__.__name__ == 'BatchOutput':
        return get_device(obj.pred)
    else:
        raise ValueError(f'{obj.__class__.__name__} is not a valid object')

class Device:
    '''cpu / cuda / mps device , callable'''
    def __init__(self , device : torch.device | None = None) -> None:
        if device is None: 
            device = use_device
        self.device = device
    def __repr__(self): return str(self.device)
    def __call__(self, obj): return send_to(obj , self.device)

    @property
    def is_cuda(self): return self.device.type == 'cuda'
    @property
    def is_mps(self): return self.device.type == 'mps'
    @property
    def is_cpu(self): return self.device.type == 'cpu'
        
    def cpu(self , x): return send_to(x , 'cpu')
    def cuda(self , x): return send_to(x , 'cuda')
    def mps(self , x): return send_to(x , 'mps')
    
    def print(self):
        if self.is_cuda:
            print(f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
                  f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G')
        elif self.is_mps:
            print(f'Allocated {torch.mps.current_allocated_memory() / 1024**3:.1f}G, '+\
                  f'Driver Allocated {torch.mps.driver_allocated_memory() / 1024**3:.1f}G')
        else:
            print(f'Not using cuda or mps {self.device}')
        
class MemoryPrinter:
    def __repr__(self) -> str:
        return 'Used: {:.2f}G; Free {:.2f}G'.format(
            float(psutil.virtual_memory().used)/1024**3,
            float(psutil.virtual_memory().free)/1024**3)
    def print(self): print(self.__repr__())
        
        