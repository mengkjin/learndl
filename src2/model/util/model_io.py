import torch

from dataclasses import dataclass
from pathlib import Path
from torch import Tensor , nn
from typing import Any , Optional

from .batch_io import BatchOutput
from ...algo.boost import GeneralBooster
from ...basic import ModelPath
    
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

    def reset(self):
        self.state_dict = None
        self.booster_head = None
        self.booster_dict = None
        self.aggregator_dict = None

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