from dataclasses import dataclass
from torch import Tensor
from typing import Any , Literal

@dataclass
class BatchOutput:
    outputs : Tensor | tuple | list
    @property
    def pred(self) -> Tensor:
        return self.outputs[0] if isinstance(self.outputs , (list , tuple)) else self.outputs
    @property
    def hidden(self) -> Tensor | None:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None
    @classmethod
    def empty(cls): return cls(Tensor().requires_grad_())

@dataclass
class BatchData:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : Tensor | tuple[Tensor] | list[Tensor]
    y       : Tensor 
    w       : Tensor | None
    i       : Tensor 
    valid   : Tensor 
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: self.x = self.x[0]
    def to(self , device = None): return self.__class__(**{k:self.send_to(v , device) for k,v in self.__dict__.items()})
    def cpu(self):  return self.__class__(**{k:self.send_to(v , 'cpu') for k,v in self.__dict__.items()})
    def cuda(self): return self.__class__(**{k:self.send_to(v , 'cuda') for k,v in self.__dict__.items()})
    @property
    def is_empty(self): return len(self.y) == 0
    @classmethod
    def send_to(cls , obj , des : Any | Literal['cpu' , 'cuda']) -> Any:
        if obj is None: return None
        elif isinstance(obj , Tensor):
            if des == 'cpu': return obj.cpu()
            elif des == 'cuda': return obj.cuda()
            elif callable(des): return des(obj) 
            else: return obj.to(des)
        elif isinstance(obj , (list , tuple)):
            return type(obj)([cls.send_to(o , des) for o in obj])
        else: raise TypeError(obj)

class BaseCallBack:
    def __init__(self) -> None:
        pass