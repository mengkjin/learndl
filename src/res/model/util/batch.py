import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from dataclasses import dataclass , field
from typing import Any

from src.basic.util.device import Device , send_to , get_device

def _object_shape(obj : Any) -> Any:
    if obj is None: 
        return None
    elif isinstance(obj , torch.Tensor | np.ndarray): 
        return obj.shape
    elif isinstance(obj , (list , tuple)): 
        return tuple([_object_shape(x) for x in obj])
    else: 
        return type(obj)

@dataclass(slots=True)
class BatchData:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : torch.Tensor | tuple[torch.Tensor,...] | list[torch.Tensor]
    y       : torch.Tensor 
    w       : torch.Tensor | None
    i       : torch.Tensor 
    valid   : torch.Tensor
    kwargs  : dict[str,Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: 
            self.x = self.x[0]
        assert self.y is not None , 'y must not be None'
        assert self.i is not None , 'i must not be None'
        assert self.valid is not None , 'valid must not be None'
    def to(self , device = None): 
        if device is None: 
            return self
        else:
            if isinstance(device , Device): 
                device = device.device
            inputs = {name:send_to(getattr(self , name) , device) for name in self.__slots__}
            return BatchData(**inputs)
        
    def cpu(self):  
        return self.to('cpu')
    def cuda(self): 
        return self.to('cuda')
    def mps(self): 
        return self.to('mps')
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, key) -> Any:
        if key == 'x': 
            return self.x
        elif key == 'y': 
            return self.y
        elif key == 'w': 
            return self.w
        elif key == 'i': 
            return self.i
        elif key == 'valid': 
            return self.valid
        else: 
            return self.kwargs[key]
    def keys(self):
        return ['x' , 'y' , 'w' , 'i' , 'valid' , *self.kwargs.keys()]
    def items(self):
        return {k:self[k] for k in self.keys()}
    
    @property
    def is_empty(self): return len(self.y) == 0
    @property
    def device(self): return self.y.device
    @property
    def shape(self): 
        return {k:_object_shape(self[k]) for k in self.keys()}
    @classmethod
    def random(cls , batch_size = 2 , seq_len = 30 , n_inputs = 6 , predict_steps = 1):

        x = torch.rand(batch_size , seq_len , n_inputs)

        y = torch.rand(batch_size , predict_steps)
        w = None
        i = torch.Tensor([])
        v = torch.Tensor([])
        return cls(x , y , w , i , v)
    @classmethod
    def concat(cls , *batch_datas):
        assert len(batch_datas) > 0
        
        x , y , w , i , v , kwargs = [] , [] , [] , [] , [] , []
        for bd in batch_datas:
            assert isinstance(bd , cls) , type(bd)
            x.append(bd.x)
            y.append(bd.y)
            w.append(bd.w)
            i.append(bd.i)
            v.append(bd.valid)
            kwargs.append(bd.kwargs)
        assert all([len(kwg) == 0 for kwg in kwargs]) , [kwg.keys() for kwg in kwargs]
        if isinstance(x[0] , torch.Tensor):
            x = torch.concat(x)
        else:
            assert all([len(xx) == len(x[0]) for xx in x]) , [len(xx) for xx in x]
            x = type(x[0])([torch.concat([xx[j] for xx in x]) for j in range(len(x[0]))])
        y = torch.concat(y)
        assert all([type(ww) is type(w[0]) for ww in w]) , [type(ww) for ww in w]
        w = None if w[0] is None else torch.concat(w)
        i = torch.concat(i)
        v = torch.concat(v)
        return cls(x , y , w , i , v)

@dataclass
class BatchMetric:
    score   : torch.Tensor | float = 0.
    losses  : dict[str,torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.score , torch.Tensor): 
            self.score = self.score.item()
        self.loss = torch.Tensor([0])
        for value in self.losses.values(): 
            self.loss = self.loss.to(value) + value

    @property
    def loss_item(self) -> float: return self.loss.item()

    def set_score(self , score : torch.Tensor | float): 
        self.score = score.item() if isinstance(score , torch.Tensor) else score

    def add_loss(self , key : str , value : torch.Tensor):
        assert key not in self.losses.keys() , (key , self.losses.keys())
        self.loss = self.loss.to(value) + value
        self.losses[key] = value

    def add_losses(self , losses : dict[str,torch.Tensor] , prefix : str | tuple[str,...] | None = None):
        if prefix is not None:
            losses = {key:value for key , value in losses.items() if key.startswith(prefix)}
        for key , value in losses.items():
            self.add_loss(key , value)

@dataclass(slots=True)
class BatchOutput:
    outputs : torch.Tensor | tuple | list | None = None
    def __len__(self): return len(self.pred)
    @property
    def device(self): return self.pred.device
    @property
    def pred(self) -> torch.Tensor:
        if self.outputs is None: 
            return torch.Tensor(size=(0,1)).requires_grad_()
        output = self.outputs[0] if isinstance(self.outputs , (list , tuple)) else self.outputs
        if output.ndim == 1: 
            output = output.unsqueeze(1)
        assert output.ndim == 2 , output.ndim
        return output
    @property
    def other(self) -> dict[str,Any]:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            assert isinstance(self.outputs[1] , dict) , type(self.outputs[1])
            return self.outputs[1]
        else:
            return {}
    
    @property
    def hidden(self) -> torch.Tensor: return self.other['hidden']
        
    def override_pred(self , pred : torch.Tensor | None):
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
    
    def pred_df(self , secid : np.ndarray | Any , date : np.ndarray | Any , narrow_df = False , 
                colnames : str | list | None = None , **kwargs):
        pred = self.pred.cpu().numpy()
        if pred.ndim == 1: 
            pred = pred[:,None]
        assert pred.ndim == 2 , pred.shape

        if colnames is None: 
            columns = [f'pred.{i}' for i in range(pred.shape[-1])]
        elif isinstance(colnames , str): 
            columns = [colnames]
        else: 
            columns = colnames
        assert pred.shape[-1] == len(columns) , (pred.shape , columns)

        df = pd.DataFrame({'secid' : secid , 'date' : date , **{col:pred[:,i] for i,col in enumerate(columns)}})
        if isinstance(colnames , str):
            assert pred.shape[-1] == 1 , (pred.shape , colnames)
            df = df.rename(columns={'pred.0':colnames})
        if narrow_df: 
            df = df.melt(['secid','date'] , var_name='feature')
        return df.assign(**kwargs)
        
    def hidden_df(self , secid : np.ndarray , date : np.ndarray , narrow_df = False ,
                  colnames : str | list | None = None , **kwargs):
        '''kwargs will be used in df.assign(**kwargs)'''
        full_hidden : torch.Tensor | Any = self.other['hidden']
        full_hidden = full_hidden.cpu().numpy()

        assert full_hidden.ndim == 2 , full_hidden.shape

        if colnames is None: 
            columns = [f'hidden.{i}' for i in range(full_hidden.shape[-1])]
        elif isinstance(colnames , str): 
            columns = [colnames]
        else: 
            columns = colnames
        assert full_hidden.shape[-1] == len(columns) , (full_hidden.shape , columns)

        df = pd.DataFrame({'secid' : secid , 'date' : date , **{col:full_hidden[:,i] for i,col in enumerate(columns)}})
        if narrow_df: 
            df = df.melt(['secid','date'] , var_name='feature')
        return df.assign(**kwargs)
    
    def __getitem__(self, key) -> Any:
        if key == 'pred': 
            return self.pred
        else: 
            return self.other[key]

    def keys(self):
        return ['pred' , *self.other.keys()]
    
    def items(self):
        return {k:self[k] for k in self.keys()}

    @property
    def shape(self):
        return {k:_object_shape(self[k]) for k in self.keys()}
    
    @classmethod
    def nn_module(cls , module : nn.Module , inputs : Any | BatchData , **kwargs):
        if isinstance(inputs , BatchData): 
            inputs = inputs.x
        device0 = get_device(module)
        device1 = get_device(inputs)
        assert device0 == device1 , (device0 , device1)
        return cls(module(inputs ,  **kwargs))