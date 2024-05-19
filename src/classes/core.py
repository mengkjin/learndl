import numpy as np
import os , torch
from dataclasses import dataclass , field
from torch import Tensor
from typing import Any , Literal , Optional

@dataclass(slots=True)
class BatchData:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : Tensor | tuple[Tensor] | list[Tensor]
    y       : Tensor 
    w       : Tensor | None
    i       : Tensor 
    valid   : Tensor
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: self.x = self.x[0]
    def to(self , device = None): 
        return self.__class__(
            x = self.send_to(self.x , device) , 
            y = self.send_to(self.y , device) ,
            w = self.send_to(self.w , device) ,
            i = self.send_to(self.i , device) ,
            valid = self.send_to(self.valid , device))
    def cpu(self):  return self.to('cpu')
    def cuda(self): return self.to('cuda')
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

@dataclass(slots=True)
class BatchMetric:
    loss      : Tensor = Tensor([0.])
    score     : float = 0.
    penalty   : Tensor | float = 0.
    losses    : Tensor = Tensor([0.])

    @property
    def loss_item(self): return self.loss.item()

@dataclass(slots=True)
class MetricList:
    name : str
    type : str
    values : list[Any] = field(default_factory=list) 

    def __post_init__(self): assert self.type in ['loss' , 'score']
    def record(self , metrics): self.values.append(metrics.loss_item if self.type == 'loss' else metrics.score)
    def last(self): self.values[-1]
    def mean(self): return np.mean(self.values)
    def any_nan(self): return np.isnan(self.values).any()

@dataclass(slots=True)
class BatchOutput:
    outputs : Optional[Tensor | tuple | list] = None

    @property
    def pred(self) -> Tensor:
        if self.outputs is None:
            return Tensor().requires_grad_()
        elif isinstance(self.outputs , (list , tuple)):
            return self.outputs[0]
        else:
            return self.outputs
    @property
    def hidden(self) -> Optional[Tensor]:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None
    def override_pred(self , pred : Tensor):
        assert self.outputs is not None
        assert len(pred) == len(self.pred) , (pred.shape , self.pred.shape)
        pred = pred.reshape(*self.pred.shape)
        if isinstance(self.outputs , (list , tuple)):
            self.outputs = [pred , *self.outputs[1:]]
        else:
            self.outputs = pred
        return self
    
@dataclass(slots=True)
class ModelDict:
    state_dict : dict[str,Tensor]
    lgbm_string : Optional[str] = None
    
    def save(self , path : str):
        for key in self.__slots__:
            value = getattr(self , key)
            if value is not None: 
                os.makedirs(os.path.dirname(path) , exist_ok=True)
                torch.save(value , path.format(key))

@dataclass
class ModelFile:
    model_path : str
    def __getitem__(self , key): return self.load(key)
    def path(self , key): return f'{self.model_path}/{key}.pt'
    def load(self , key : str) -> Any:
        assert key in ModelDict.__slots__
        path = self.path(key)
        return torch.load(path , map_location='cpu') if os.path.exists(path) else None
    def exists(self) -> bool: return os.path.exists(self.path('state_dict'))
    