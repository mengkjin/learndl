import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass , field
from torch import Tensor
from typing import Any , Literal , Optional

from ....algo.boost import BoosterInput

@dataclass(slots=True)
class BatchData:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : Tensor | tuple[Tensor,...] | list[Tensor]
    y       : Tensor 
    w       : Tensor | None
    i       : Tensor 
    valid   : Tensor
    kwargs  : dict[str,Any] = field(default_factory=dict)
    boost   : BoosterInput | None = None
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: self.x = self.x[0]
    def to(self , device = None): 
        return self.__class__(
            x = self.send_to(self.x , device) , 
            y = self.send_to(self.y , device) ,
            w = self.send_to(self.w , device) ,
            i = self.send_to(self.i , device) ,
            valid = self.send_to(self.valid , device) ,
            kwargs = self.send_to(self.kwargs , device))
    def cpu(self):  return self.to('cpu')
    def cuda(self): return self.to('cuda')
    @property
    def is_empty(self): return len(self.y) == 0
    @classmethod
    def send_to(cls , obj , des : Any | Literal['cpu' , 'cuda']) -> Any:
        if isinstance(obj , Tensor):
            if des == 'cpu': return obj.cpu()
            elif des == 'cuda': return obj.cuda()
            elif callable(des): return des(obj) 
            else: return obj.to(des)
        elif isinstance(obj , (list , tuple)):
            return type(obj)([cls.send_to(o , des) for o in obj])
        elif isinstance(obj , dict):
            return {k:cls.send_to(o , des) for k,o in obj.items()}
        else: return obj
    @property
    def device(self): return self.y.device
    @classmethod
    def random(cls , batch_size = 2 , seq_len = 30 , n_inputs = 6 , predict_steps = 1):
        patch_len = 3
        stride = 2
        mask_ratio = 0.4
        d_model = 16

        x = torch.rand(batch_size , seq_len , n_inputs)

        y = torch.rand(batch_size , predict_steps)
        w = None
        i = torch.Tensor([])
        v = torch.Tensor([])
        return cls(x , y , w , i , v)

@dataclass
class BatchMetric:
    score   : Tensor | float = 0.
    losses  : dict[str,Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.score , Tensor): self.score = self.score.item()
        self.loss = Tensor([0])
        for value in self.losses.values(): 
            self.loss = self.loss.to(value) + value

    @property
    def loss_item(self) -> float: return self.loss.item()

    def add_loss(self , key : str , value : Tensor):
        assert key not in self.losses.keys() , (key , self.losses.keys())
        self.loss = self.loss.to(value) + value
        self.losses[key] = value

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
    def __len__(self): return len(self.pred)
    @property
    def device(self): return self.pred.device
    @property
    def pred(self) -> Tensor:
        if self.outputs is None:
            return Tensor(size=(0,1)).requires_grad_()
        elif isinstance(self.outputs , (list , tuple)):
            return self.outputs[0]
        else:
            return self.outputs
    @property
    def other(self) -> dict[str,Any]:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            assert isinstance(self.outputs[1] , dict)
            return self.outputs[1]
        else:
            return {}
    
    @property
    def hidden(self) -> Tensor: return self.other['hidden']
        
    def override_pred(self , pred : Optional[Tensor]):
        assert self.outputs is not None
        assert pred is not None
        raw_pred = self.pred
        assert len(pred) == len(raw_pred) , (pred.shape , raw_pred.shape)
        pred = pred.reshape(*raw_pred.shape).to(raw_pred)
        if isinstance(self.outputs , (list , tuple)):
            self.outputs = [pred , *self.outputs[1:]]
        else:
            self.outputs = pred
        return self
    
    def pred_df(self , secid : np.ndarray , date : np.ndarray , narrow_df = False):
        pred = self.pred.cpu().numpy()
        if pred.ndim == 1: pred = pred[:,None]
        assert pred.ndim == 2 , pred.shape
        df = pd.DataFrame({'secid' : secid , 'date' : date , **{f'pred.{i}':pred[:,i] for i in range(pred.shape[-1])}})
        if narrow_df:
            df = df.melt(['secid','date'] , var_name='feature')
        return df
        
    def hidden_df(self , batch_data : BatchData , y_secid : np.ndarray , y_date : np.ndarray , narrow_df = False):
        full_hidden : Tensor = self.other['hidden']
        full_hidden = full_hidden.cpu().numpy()

        ij = batch_data.i.cpu()
        secid = y_secid[ij[:,0]]
        date  = y_date[ij[:,1]]

        assert full_hidden.ndim == 2 , full_hidden.shape

        df = pd.DataFrame({'secid' : secid , 'date' : date , 
                           **{f'hidden.{i}':full_hidden[:,i] for i in range(full_hidden.shape[-1])}})
        if narrow_df: df = df.melt(['secid','date'] , var_name='feature')
        return df