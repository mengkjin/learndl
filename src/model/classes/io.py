import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass , field
from pathlib import Path
from torch import Tensor , nn
from typing import Any , Literal , Optional

from ..boost import GeneralBooster
from ...basic import ModelPath

@dataclass(slots=True)
class BatchData:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : Tensor | tuple[Tensor,...] | list[Tensor]
    y       : Tensor 
    w       : Tensor | None
    i       : Tensor 
    valid   : Tensor
    kwargs  : dict[str,Any] = field(default_factory=dict)
    
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
    
@dataclass(slots=True)
class ModelInstance:
    net  : Optional[nn.Module] = None
    booster_head : Optional[GeneralBooster] = None
    booster      : Optional[GeneralBooster] = None
    aggregator   : Optional[GeneralBooster] = None

    def __call__(self , x : Tensor | tuple[Tensor,...] | list[Tensor] , *args , **kwargs):
        return self.forward(x , *args , **kwargs)

    def forward(self , x : Tensor | tuple[Tensor,...] | list[Tensor]  , *args , **kwargs):
        if self.net is not None:
            assert self.booster is None and self.aggregator is None
            out = BatchOutput(self.net(x , *args , **kwargs))
            if self.booster_head is not None:
                return self.booster_head.forward(out.hidden)
            else:
                return out.pred
        else:
            assert self.booster_head is None
            x = x if isinstance(x , Tensor) else x[0]
            if self.booster is not None:
                assert self.aggregator is None
                return self.booster.forward(x)
            else:
                assert self.aggregator is not None
                return self.aggregator.forward(x)
            
    def eval(self):
        if self.net is not None: self.net.eval()
        return self

    def train(self):
        if self.net is not None: self.net.train()
        return self

@dataclass(slots=True)
class ModelDict:
    state_dict  : Optional[dict[str,Tensor]] = None
    booster_head : Optional[dict[str,Any]] = None
    booster_dict : Optional[dict[str,Any]] = None
    aggregator_dict : Optional[dict[str,Any]] = None

    def save(self , path : str | Path , stack = False):
        if isinstance(path , str): path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
        for key in self.__slots__:
            if (value := getattr(self , key)) is not None:
                torch.save(value , path.joinpath(f'{key}.stack.pt' if stack else f'{key}.pt'))

    @property
    def legal(self):
        if self.state_dict is not None:
            assert self.booster_dict is None and self.aggregator_dict is None
        else:
            assert self.booster_head is None
            if self.booster_dict is not None:
                assert self.aggregator_dict is None
            else:
                assert self.aggregator_dict is not None
        return True

    def model_instance(self , base_net : nn.Module | Any = None , device = None , **kwargs):
        assert self.legal
        if self.state_dict is not None:
            base_net.load_state_dict(self.state_dict)
            net = device(base_net) if callable(device) else base_net.to(device)
        else:
            net = None
        booster_head = GeneralBooster.from_dict(self.booster_head) if self.booster_head is not None else None
        booster      = GeneralBooster.from_dict(self.booster_dict) if self.booster_dict is not None else None
        aggregator   = GeneralBooster.from_dict(self.aggregator_dict) if self.aggregator_dict is not None else None
        return ModelInstance(net = net , booster_head=booster_head , booster=booster , aggregator=aggregator)

@dataclass
class ModelFile:
    model_path : Path
    def __getitem__(self , key): return self.load(key)
    def path(self , key): return self.model_path.joinpath(f'{key}.pt')
    def load(self , key : str) -> Any:
        assert key in ModelDict.__slots__ , (key , ModelDict.__slots__)
        path = self.path(key)
        return torch.load(path , map_location='cpu') if path.exists() else None
    def exists(self) -> bool: 
        return any([self.path(key).exists() for key in ModelDict.__slots__])
    def model_dict(self):
        return ModelDict(**{key:self.load(key) for key in ModelDict.__slots__})
    def model_instance(self , base_net : nn.Module | Any = None , device = None , **kwargs):
        return self.model_dict().model_instance(base_net , device , **kwargs)
    

class SingleModelInterface:
    def __init__(self , path : Path) -> None:
        self.path = path
        
    def __getitem__(self , key): return self.load(key)

    def load(self , key : str) -> Any:
        assert key in ModelDict.__slots__ , (key , ModelDict.__slots__)
        path = self.path.joinpath(f'{key}.pt')
        return torch.load(path , map_location='cpu') if path.exists() else None
    def exists(self) -> bool: 
        return any([self.path.joinpath(f'{key}.pt').exists() for key in ModelDict.__slots__])
    def model_dict(self):
        return ModelDict(**{key:self.load(key) for key in ModelDict.__slots__})
    def model_instance(self , base_net : nn.Module | Any = None , device = None , **kwargs):
        return self.model_dict().model_instance(base_net , device , **kwargs)

class ModelInterface:
    def __init__(self , model_path : Path | ModelPath | str) -> None:
        self.model_path = ModelPath(model_path)

    def single(self , model_date , model_num , model_type = 'best'):
        path = self.model_path.full_path(model_num = model_num , model_date = model_date , model_type=model_type)
        return SingleModelInterface(path)