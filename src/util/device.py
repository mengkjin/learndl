import psutil , torch
from typing import Any

use_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): print(f'Use device name: ' + torch.cuda.get_device_name(0))
        
class Device:
    '''cpu / cuda device , callable'''
    def __init__(self , device = None) -> None:
        if device is None: device = use_device
        self.device = device
    def __repr__(self): return str(self.device)
    def __call__(self, obj): return self.send_to(obj , self.device)
    
    @classmethod
    def send_to(cls , x , device = None) -> Any:
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

    def print(self):
        print(f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
              f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G')
        
class MemoryPrinter:
    def __repr__(self) -> str:
        return 'Used: {:.2f}G; Free {:.2f}G'.format(
            float(psutil.virtual_memory().used)/1024**3,
            float(psutil.virtual_memory().free)/1024**3)
    def print(self): print(self.__repr__())
        
        