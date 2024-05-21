import numpy as np
import torch
from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from inspect import currentframe
from torch import Tensor
from typing import Any , Callable , final , Iterable , Iterator , Literal , Optional

from .core import BatchData , BatchOutput

@dataclass
class EndStatus:
    name  : str
    epoch : int # epoch of trigger
    
@dataclass
class EndofLoop:
    max_epoch : int = 200
    status : list[EndStatus] = field(default_factory=list)

    def __post_init__(self) -> None: ...
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
    stage   : Literal['data' , 'fit' , 'test'] = 'data'
    dataset : Literal['train' , 'validation' , 'test'] = 'train'
    epoch   : int = -1
    attempt : int = 0
    round   : int = 0
    model_num  : int = -1
    model_date : int = -1
    model_type : str = 'best'
    end_of_loop  : EndofLoop = field(default_factory=EndofLoop)
    epoch_event  : list[str] = field(default_factory=list)
    best_attempt_metric : Any = None

    def stage_data(self): self.stage = 'data'
    def stage_fit(self):  self.stage = 'fit'
    def stage_test(self): self.stage = 'test'
    def dataset_train(self): self.dataset = 'train'
    def dataset_validation(self): self.dataset = 'validation'
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
        self.end_of_loop = EndofLoop(self.max_epoch)
        self.epoch_event = []

        self.add_event(event)
        if event == 'new_attempt': self.attempt += 1

    def add_event(self , event : Optional[str]):
        if event: self.epoch_event.append(event)

class BaseCB:
    def __init__(self , model_module , with_cb) -> None:
        self.model_module : BaseModelModule = model_module
        self.with_cb = with_cb
        self.__hook_stack = []
        self._assert_validity()
    def _print_info(self):
        args = {k:v for k,v in getattr(currentframe() , 'f_back').f_locals.items() if k not in ['self','model_module'] and not k.startswith('_')}
        info = f'Callback : {self.__class__.__name__}' + '({})'.format(','.join([f'{k}={v}' for k,v in args.items()])) 
        if self.__class__.__doc__: info += f' , {self.__class__.__doc__}'
        print(info)
    def __call__(self , hook : Any = None): 
        if hook is None : hook = self.trace_hook_name
        if isinstance(hook , str): return self.hook_proceed(hook)
        elif callable(hook): return self.hook_wrapper(hook)
        else: raise TypeError(hook)
    def hook_wrapper(self , hook : Callable):
        hook_name = hook.__name__
        def wrapper_normal():
            hook()
            self.hook_proceed(hook_name)
        def wrapper_with():
            self.at_enter(hook.__name__)
            hook()
            self.at_exit(hook.__name__)
        return wrapper_with if self.with_cb else wrapper_normal
    def hook_proceed(self , hook_name): getattr(self , hook_name)()
    def at_enter(self , hook_name):  ...
    def at_exit(self , hook_name):  self.hook_proceed(hook_name)
    def __enter__(self): 
        self.__hook_stack.append(self.trace_hook_name)
        self.at_enter(self.__hook_stack[-1])
    def __exit__(self , *args): self.at_exit(self.__hook_stack.pop())
    @property
    def module(self): return self.model_module
    def on_configure_model(self): ...
    def on_summarize_model(self): ...
    def on_data_end(self): ... 
    def on_data_start(self): ... 
    def on_after_backward(self): ... 
    def on_after_fit_epoch(self): ... 
    def on_before_backward(self): ... 
    def on_before_save_model(self): ... 
    def on_before_fit_epoch_end(self): ... 
    def on_fit_end(self): ... 
    def on_fit_epoch_end(self): ... 
    def on_fit_epoch_start(self): ... 
    def on_fit_model_end(self): ... 
    def on_fit_model_start(self): ... 
    def on_fit_start(self): ...
    def on_test_batch(self): ... 
    def on_test_batch_end(self): ...
    def on_test_batch_start(self): ... 
    def on_test_end(self): ... 
    def on_test_model_end(self): ... 
    def on_test_model_start(self): ... 
    def on_test_model_type_end(self): ... 
    def on_test_model_type_start(self): ... 
    def on_test_start(self): ... 
    def on_train_batch(self): ... 
    def on_train_batch_end(self): ... 
    def on_train_batch_start(self): ... 
    def on_train_epoch_end(self): ... 
    def on_train_epoch_start(self): ... 
    def on_validation_batch(self): ... 
    def on_validation_batch_end(self): ... 
    def on_validation_batch_start(self): ...
    def on_validation_epoch_end(self): ...
    def on_validation_epoch_start(self): ...
    @classmethod
    def _possible_hooks(cls):
        return [x for x in dir(cls) if cls._possible_hook(x)]
    @classmethod
    def _possible_hook(cls , name : str):
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
    def register_setup(self) -> None: ...
    @abstractmethod
    def register_update(self) -> None: ...

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
        self.y_secid : Any
        self.y_date : Any
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
        return batch.to(getattr(self , 'device' , None) if device is None else device)
    def on_after_batch_transfer(self , batch : BatchData , dataloader_idx = None): return batch

class BaseModelModule(ABC):
    '''run through the whole process of training'''
    @final
    def __init__(self , **kwargs):
        self.init_config(**kwargs)
        self.init_utilities(**kwargs)
        self.init_data(**kwargs)
        self.status = TrainerStatus(getattr(self.config , 'max_epoch'))
        [setattr(self , x , self.callbacks(getattr(self , x))) for x in dir(self) if BaseCB._possible_hook(x)]

    @abstractmethod
    def batch_forward(self) -> None: 
        '''forward of batch_data'''
        self.batch_output = self(self.batch_data)
    @abstractmethod
    def batch_metrics(self) -> None: 
        '''calculate and collect of batch_data'''
        ...
    @abstractmethod
    def batch_backward(self) -> None: 
        '''backward of batch loss'''
        ...
    @abstractmethod
    def init_config(self , **kwargs) -> None:
        '''initialized configuration'''
        self.config = kwargs['config']
        self.stage_queue = ['data' , 'fit' , 'test']
    @abstractmethod
    def init_utilities(self , **kwargs): 
        '''initialized all relevant utilities'''
        self.logger     = kwargs['logger']
        self.checkpoint = kwargs['checkpoint']
        self.deposition = kwargs['deposition']
        self.metrics    = kwargs['metrics']
        self.callbacks  = kwargs['callbacks']
        self.device     = kwargs['device']
        self.model      = kwargs['model']
        self.dataloader : Iterable[BatchData] | Iterator[BatchData] = kwargs['device']
    @abstractmethod
    def init_data(self , **kwargs): 
        '''initialized data_module'''
        self.data : BaseDataModule = kwargs['data']
    @abstractmethod
    def save_model(self) -> None: 
        '''save self.net to somewhere'''
    @abstractmethod
    def stack_model(self) -> None: 
        '''temporaly save self.net to somewhere'''
    @abstractmethod
    def load_model(self , *args , **kwargs) -> None: 
        '''load self.net to somewhere'''
        self.net = torch.nn.Module()
        self.optimizer = args[0]
    @property
    @abstractmethod
    
    def model_param(self) -> dict:  '''current model param'''
    @property
    @abstractmethod
    def model_iter(self) -> Iterator[tuple[int,int]]: '''iter of model_date and model_num , considering resume_training'''
    @property
    @abstractmethod
    def model_types(self) -> list[str]: '''iter of model_type'''
    
    def __call__(self , input):
        if not isinstance(input , BatchData):
            return BatchOutput(self.net(input))
        elif input.is_empty:
            return BatchOutput()
        else:
            return BatchOutput(self.net(input.x))

    def main_process(self):
        '''Main stage of data & fit & test'''
        self.on_configure_model()
        for self.stage in self.stage_queue: 
            getattr(self , f'stage_{self.stage}')()
        self.on_summarize_model()

    def stage_data(self):
        '''stage of loading model data'''
        self.status.stage_data()
        self.on_data_start()
        self.data.load_data()
        self.on_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.status.stage_fit()
        self.on_fit_start()
        for self.status.model_date , self.status.model_num in self.model_iter:
            self.fit_model()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.status.stage_test()
        self.on_test_start()
        for self.status.model_date , self.status.model_num in self.model_iter:
            self.test_model()
        self.on_test_end()

    def fit_model(self):
        self.status.fit_model_start()
        self.on_fit_model_start()
        while not self.status.end_of_loop:
            self.status.fit_epoch_start()
            self.on_fit_epoch_start()

            self.status.dataset_train()
            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()

            self.status.dataset_validation()
            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()

            self.on_before_fit_epoch_end()
            self.status.fit_epoch_end()
            self.on_fit_epoch_end()
        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        for self.status.model_type in self.model_types:
            self.status.dataset_test()
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    @property
    def penalty_kwargs(self): return {}
    def on_configure_model(self): ... 
    def on_summarize_model(self): ...
    def on_data_start(self): ...
    def on_data_end(self): ...
    def on_fit_start(self): ...
    def on_fit_end(self): ...
    def on_test_start(self): ...
    def on_test_end(self): ...
    def on_fit_model_start(self): ...
    def on_fit_model_end(self): ...
    def on_train_batch(self):
        self.batch_forward()
        self.batch_metrics()
        self.batch_backward()
    def on_validation_batch(self):
        self.batch_forward()
        self.batch_metrics()
    def on_test_batch(self):
        self.batch_forward()
        self.batch_metrics()
    def on_fit_epoch_start(self): ...
    def on_before_fit_epoch_end(self): ...
    def on_fit_epoch_end(self): ...
    def on_train_epoch_start(self):
        self.net.train()
        torch.set_grad_enabled(True)
    def on_train_epoch_end(self): ...
    def on_validation_epoch_start(self):
        self.net.eval()
        torch.set_grad_enabled(False)
    def on_validation_epoch_end(self):
        torch.set_grad_enabled(True)
    def on_test_model_start(self):
        self.net.eval()
        torch.set_grad_enabled(False)
    def on_test_model_end(self):
        torch.set_grad_enabled(True)
    def on_test_model_type_start(self): ...
    def on_test_model_type_end(self): ...
    def on_train_batch_start(self): ...
    def on_train_batch_end(self): ...
    def on_validation_batch_start(self): ...
    def on_validation_batch_end(self): ...
    def on_test_batch_start(self): ...
    def on_test_batch_end(self): ...
    def on_before_backward(self): ...
    def on_after_backward(self): ...
    def on_before_save_model(self): ...
    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)