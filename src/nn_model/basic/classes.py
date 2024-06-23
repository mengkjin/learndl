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
    @property
    def device(self): return self.pred.device
    @property
    def pred(self) -> Tensor:
        if self.outputs is None:
            return Tensor().requires_grad_()
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
    
@dataclass(slots=True)
class ModelDict:
    state_dict  : Optional[dict[str,Tensor]] = None
    booster_str : Optional[str] = None
    
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
        assert key in ModelDict.__slots__ , (key , ModelDict.__slots__)
        path = self.path(key)
        return torch.load(path , map_location='cpu') if os.path.exists(path) else None
    def exists(self) -> bool: 
        return any([os.path.exists(self.path(key)) for key in ModelDict.__slots__])
    
@dataclass
class TrainerStatus:
    max_epoch : int = 200
    stage   : Literal['data' , 'fit' , 'test'] = 'data'
    dataset : Literal['train' , 'valid' , 'test'] = 'train'
    epoch   : int = -1
    attempt : int = 0
    round   : int = 0
    model_num  : int = -1
    model_date : int = -1
    model_type : str = 'best'
    epoch_event  : list[str] = field(default_factory=list)
    best_attempt_metric : Any = None

    def __post_init__(self):
        self.end_of_loop = self.EndofLoop(self.max_epoch)

    def stage_data(self): self.stage = 'data'
    def stage_fit(self):  self.stage = 'fit'
    def stage_test(self): self.stage = 'test'
    def dataset_train(self): self.dataset = 'train'
    def dataset_validation(self): self.dataset = 'valid'
    def dataset_test(self): self.dataset = 'test'
    def fit_model_start(self):
        self.attempt = -1
        self.best_attempt_metric = None
        self.new_attempt()
    def fit_epoch_start(self):
        self.epoch   += 1
        self.epoch_event = []
    def fit_epoch_end(self):
        self.end_of_loop.loop_end(self.epoch)
    def new_attempt(self , event : Literal['new_attempt' , 'nanloss'] = 'new_attempt'):
        self.epoch   = -1
        self.round   = 0
        self.end_of_loop = self.EndofLoop(self.max_epoch)
        self.epoch_event = []

        self.add_event(event)
        if event == 'new_attempt': self.attempt += 1

    def add_event(self , event : Optional[str]):
        if event: self.epoch_event.append(event)

    @dataclass
    class EndofLoop:
        max_epoch : int = 200
        status : list['EndStatus'] = field(default_factory=list)

        @dataclass
        class EndStatus:
            name  : str
            epoch : int # epoch of trigger

        def __post_init__(self) -> None: ...
        def __bool__(self): return len(self.status) > 0
        def new_loop(self): self.status = []
        def loop_end(self , epoch):
            if epoch >= self.max_epoch: self.add_status('Max Epoch' , epoch)
        def add_status(self , status : str , epoch : int): 
            self.status.append(self.EndStatus(status , epoch))
        @property
        def end_epochs(self) -> list[int]:
            return [sta.epoch for sta in self.status]
        @property
        def trigger_i(self) -> int:
            return np.argmin(self.end_epochs).item()
        @property
        def trigger_ep(self) -> int:
            return self.status[self.trigger_i].epoch
        @property
        def trigger_reason(self):
            return self.status[self.trigger_i].name