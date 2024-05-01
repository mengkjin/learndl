import numpy as np
import torch
from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from inspect import currentframe
from torch import Tensor
from typing import Any , Callable , Iterator , Literal , Optional

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
class EndStatus:
    name  : str
    epoch : int # epoch of trigger
    
@dataclass
class EndofLoop:
    max_epoch : int = 200
    status : list[EndStatus] = field(default_factory=list)

    def __post_init__(self) -> None: pass
    def __bool__(self): return len(self.status) > 0
    def new_loop(self): self.status = []
    def loop_end(self , epoch):
        if epoch >= self.max_epoch: self.add_status('Max Epoch' , epoch)
    def add_status(self , status : str , epoch : int): 
        self.status.append(EndStatus(status , epoch))
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
    
@dataclass
class TrainerStatus:
    max_epoch : int = 200
    epoch   : int = -1
    attempt : int = 0
    round   : int = 0
    model_num  : int = -1
    model_date : int = -1
    model_type : str = 'best'
    end_of_loop  : EndofLoop = field(default_factory=EndofLoop)
    epoch_event  : list[str] = field(default_factory=list)

    def new_model(self):
        self.attempt = 0
        self.new_attempt()

    def new_attempt(self):
        self.epoch   = -1
        self.round   = 0
        self.end_of_loop = EndofLoop(self.max_epoch)
        self.epoch_event = []

    def new_epoch(self):
        self.epoch   += 1
        self.epoch_event = []

    def add_event(self , event : Optional[str]):
        if event: self.epoch_event.append(event)

    def end_epoch(self):
        self.end_of_loop.loop_end(self.epoch)

class BaseCB:
    def __init__(self , model_module) -> None:
        self.__model_module : BaseModelModule = model_module
        self._assert_validity()
    def _print_info(self):
        args = {k:v for k,v in getattr(currentframe() , 'f_back').f_locals.items() if k not in ['self','model_module'] and not k.startswith('_')}
        info = f'Callback : {self.__class__.__name__}' + '({})'.format(','.join([f'{k}={v}' for k,v in args.items()])) 
        if self.__class__.__doc__: info += f' , {self.__class__.__doc__}'
        print(info)
    def __call__(self , hook_name): self.__getattribute__(hook_name)()
    @property
    def module(self): return self.__model_module
    def on_configure_model(self): pass
    def on_summarize_model(self): pass
    def on_data_end(self): pass 
    def on_data_start(self): pass 
    def on_after_backward(self): pass 
    def on_after_fit_epoch(self): pass 
    def on_before_backward(self): pass 
    def on_before_save_model(self): pass 
    def on_fit_end(self): pass 
    def on_fit_epoch_end(self): pass 
    def on_fit_epoch_start(self): pass 
    def on_fit_model_end(self): pass 
    def on_fit_model_start(self): pass 
    def on_fit_start(self): pass
    def on_test_batch(self): pass 
    def on_test_batch_end(self): pass
    def on_test_batch_start(self): pass 
    def on_test_end(self): pass 
    def on_test_model_end(self): pass 
    def on_test_model_start(self): pass 
    def on_test_model_type_end(self): pass 
    def on_test_model_type_start(self): pass 
    def on_test_start(self): pass 
    def on_train_batch(self): pass 
    def on_train_batch_end(self): pass 
    def on_train_batch_start(self): pass 
    def on_train_epoch_end(self): pass 
    def on_train_epoch_start(self): pass 
    def on_validation_batch(self): pass 
    def on_validation_batch_end(self): pass 
    def on_validation_batch_start(self): pass
    def on_validation_epoch_end(self): pass
    def on_validation_epoch_start(self): pass
    @classmethod
    def _possible_hooks(cls):
        return [x for x in dir(cls) if cls._possible_hook(x)]
    @classmethod
    def _possible_hook(cls , name):
        return name.startswith('on_') and callable(getattr(cls , name))
        # return ['self'] == [v.name for v in signature(getattr(cls , name)).parameters.values()]
    @classmethod
    def _assert_validity(cls):
        if BaseCB in cls.__bases__:
            base_hooks = BaseCB._possible_hooks()
            self_hooks = cls._possible_hooks()
            invalid_hooks = [x for x in self_hooks if x not in base_hooks]
            if invalid_hooks:
                print(f'Invalid Hooks of {cls.__name__} :' , invalid_hooks)
                print('Use _ or __ to prefix these class-methods')
                raise TypeError(cls)
    @property
    def trace_hook_name(self):
        env = getattr(currentframe() , 'f_back')
        while not env.f_code.co_name.startswith('on_'): env = getattr(env , 'f_back')
        return env.f_code.co_name
    
class BaseBuffer(ABC):
    '''dynamic buffer space for some module to use (tra), can be updated at each batch / epoch '''
    def __init__(self , key : Optional[str] = None , param : dict = {} , device : Optional[Callable] = None , always_on_device = True) -> None:
        self.key = key
        self.param = param
        self.device = device
        self.always = always_on_device
        self.contents : dict[str,Any] = {}

        self.register_setup()
        self.register_update()

    def __getitem__(self , key): return self.contents[key]
    def __setitem__(self , key , value): self.contents[key] = value
    @staticmethod
    def none_wrapper(*args, **kwargs): return {}

    def update(self , new = None):
        if new is not None: 
            if self.always and self.device is not None: new = self.device(new)
            self.contents.update(new)
        return self
    
    def get(self , keys , default = None , keep_none = True):
        if hasattr(keys , '__len__'):
            result = {k:self.contents.get(k , default) for k in keys}
            if not keep_none: result = {k:v for k,v in result.items() if v is not None}
        else:
            result = self.contents.get(keys , default)
        if not self.always and self.device is not None: result = self.device(result)
        return result

    def process(self , stage : Literal['setup' , 'update'] , data_module):
        new = getattr(self , f'{stage}_wrapper')(data_module)
        if new is not None: 
            if self.always and self.device is not None: new = self.device(new)
            self.contents.update(new)
        return self
    
    @abstractmethod
    def register_setup(self) -> None: pass
    @abstractmethod
    def register_update(self) -> None: pass

class BaseDataModule(ABC):
    '''A class to store relavant training data'''
    @abstractmethod
    def prepare_data() -> None: '''prepare all data in advance of training'''
    @abstractmethod
    def load_data() -> None: '''load prepared data at training begin'''
    @abstractmethod
    def setup(self) -> None: 
        '''create train / valid / test dataloaders'''
        self.y : Tensor
        self.buffer : BaseBuffer
    @abstractmethod
    def train_dataloader(self) -> Iterator[BatchData]: '''return train dataloaders'''
    @abstractmethod
    def val_dataloader(self) -> Iterator[BatchData]: '''return valid dataloaders'''
    @abstractmethod
    def test_dataloader(self) -> Iterator[BatchData]: '''return test dataloaders'''
    @abstractmethod
    def predict_dataloader(self) -> Iterator[BatchData]: '''return predict dataloaders'''
    def on_before_batch_transfer(self , batch : BatchData , dataloader_idx = None): return batch
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None): 
        batch.to(getattr(self , 'device' , None) if device is None else device)
    def on_after_batch_transfer(self , batch : BatchData , dataloader_idx = None): return batch

class BaseModelModule(ABC):
    '''run through the whole process of training'''
    @abstractmethod
    def __init__(self , status : TrainerStatus , net : torch.nn.Module , callbacks , data_mod):
        self.status    = status
        self.net       = net
        self.callbacks = callbacks
        assert isinstance(data_mod ,BaseDataModule)
        self.data_mod  = data_mod

    def main_process(self):
        '''Main stage of data & fit & test'''
        self.on_configure_model()
        self.stage_data()
        self.stage_fit()
        self.stage_test()
        self.on_summarize_model()

    def stage_data(self):
        '''stage of loading model data'''
        self.on_data_start()
        self.data_mod.load_data()
        self.on_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.on_fit_start()
        for self.status.model_date , self.status.model_num in self.model_iter:
            self.fit_model()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.on_test_start()
        for self.status.model_date , self.status.model_num in self.model_iter:
            self.test_model()
        self.on_test_end()

    def fit_model(self):
        self.on_fit_model_start()
        while not self.status.end_of_loop:
            self.on_fit_epoch_start()
            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()

            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()
            self.on_fit_epoch_end()

        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        for self.status.model_type in self.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    @property
    def penalty_kwargs(self): 
        '''self customed penalty kwargs , except for net , hidden and label'''
        return {}
    @abstractmethod
    def batch_forward(self) -> None: '''forward of batch_data'''
    @abstractmethod
    def batch_metrics(self) -> None: '''calculate and collect of batch_data'''
    @abstractmethod
    def batch_backward(self) -> None: '''backward of batch loss'''
    @property
    @abstractmethod
    def model_param(self) -> dict:  '''current model param'''
    @property
    @abstractmethod
    def model_iter(self) -> Iterator[tuple[int,int]]: '''iter of model_date and model_num , considering resume_training'''
    @property
    @abstractmethod
    def model_types(self) -> list[str]: '''iter of model_type'''

    def on_configure_model(self):  
        with self.callbacks: pass
    def on_summarize_model(self):
        with self.callbacks: pass
    def on_data_start(self):
        with self.callbacks: pass
    def on_data_end(self):
        with self.callbacks: pass
    def on_fit_start(self):
        with self.callbacks: pass
    def on_fit_end(self):
        with self.callbacks: pass
    def on_test_start(self): 
        with self.callbacks: pass
    def on_test_end(self): 
        with self.callbacks: pass
    def on_fit_model_start(self):
        with self.callbacks: pass
    def on_fit_model_end(self):
        with self.callbacks: pass
    def on_train_batch(self):
        with self.callbacks:
            self.batch_forward()
            self.batch_metrics()
            self.batch_backward()
    def on_validation_batch(self):
        with self.callbacks:
            self.batch_forward()
            self.batch_metrics()
    def on_test_batch(self):
        with self.callbacks:
            self.batch_forward()
            self.batch_metrics()
    def on_fit_epoch_start(self):
        with self.callbacks: pass
    
    def on_fit_epoch_end(self):
        with self.callbacks: pass
    def on_train_epoch_start(self):
        with self.callbacks:
            self.net.train()
            torch.set_grad_enabled(True)
            self.dataloader = self.data_mod.train_dataloader()
    def on_train_epoch_end(self): 
        with self.callbacks: pass
    def on_validation_epoch_start(self):
        with self.callbacks:
            self.net.eval()
            torch.set_grad_enabled(False)
            self.dataloader = self.data_mod.val_dataloader()
    def on_validation_epoch_end(self):
        with self.callbacks: 
            torch.set_grad_enabled(True)
    def on_test_model_start(self):
        with self.callbacks:
            self.net.eval()
            torch.set_grad_enabled(False)
    def on_test_model_end(self):
        with self.callbacks: 
            torch.set_grad_enabled(True)
    def on_test_model_type_start(self):
        with self.callbacks:
            self.dataloader = self.data_mod.test_dataloader()
    def on_test_model_type_end(self): 
        with self.callbacks: pass
    def on_train_batch_start(self): 
        with self.callbacks: pass
    def on_train_batch_end(self): 
        with self.callbacks: pass
    def on_validation_batch_start(self): 
        with self.callbacks: pass
    def on_validation_batch_end(self): 
        with self.callbacks: pass
    def on_test_batch_start(self): 
        with self.callbacks: pass
    def on_test_batch_end(self): 
        with self.callbacks: pass
    def on_before_backward(self): 
        with self.callbacks: pass
    def on_after_backward(self): 
        with self.callbacks: pass
    def on_before_save_model(self): 
        with self.callbacks: pass
    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)