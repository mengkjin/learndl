import numpy as np
from abc import ABC , abstractmethod
from inspect import currentframe
from logging import Logger
from torch import nn , Tensor
from typing import Any , Callable , final , Iterable , Iterator , Literal , Optional

from .io import BatchData , BatchOutput
from .status import TrainerStatus
from ...boost import BoosterInput

class BaseCB:
    def __init__(self , model_module , with_cb , turn_off = False) -> None:
        self.module : BaseTrainer = model_module
        self.with_cb : bool = with_cb
        self.turn_off : bool = turn_off
        self.__hook_stack = []
        self._assert_validity()
    def _print_info(self , depth = 0):
        frame = currentframe()
        for _ in range(depth + 1): frame = getattr(frame , 'f_back')
        args = {k:v for k,v in getattr(frame , 'f_locals').items() if k not in ['self','model_module'] and not k.startswith('_')}
        info = f'Callback : {self.__class__.__name__}' + '({})'.format(','.join([f'{k}={v}' for k,v in args.items()])) 
        if self.__class__.__doc__: info += f' , {self.__class__.__doc__}'
        print(info)
    def __call__(self , hook : Optional[str | Callable] = None) -> Callable: 
        if hook is None : hook = self.trace_hook_name
        return self.hook_wrapper(hook)
    def __enter__(self): 
        self.__hook_stack.append(self.trace_hook_name)
        self.at_enter(self.__hook_stack[-1])
    def __exit__(self , *args): self.at_exit(self.__hook_stack.pop())
    def at_enter(self , hook_name):  ...
    def at_exit(self , hook_name): getattr(self , hook_name)()
    def hook_wrapper(self , hook : str | Callable) -> Callable:
        if isinstance(hook , str):
            return getattr(self , hook)
        elif callable(hook):
            hook_name = hook.__name__
            def wrapper_normal() -> None:
                hook()
                self.at_exit(hook_name)
            def wrapper_with() -> None:
                self.at_enter(hook_name)
                hook()
                self.at_exit(hook_name)
            return wrapper_with if self.with_cb else wrapper_normal
        else:
            raise TypeError(hook)
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
    def _possible_hooks(cls) -> list[str]:
        return [x for x in dir(cls) if cls._possible_hook(x)]
    @classmethod
    def _possible_hook(cls , name : str):
        return name.startswith('on_') and callable(getattr(cls , name))
        # return ['self'] == [v.name for v in signature(getattr(cls , name)).parameters.values()]
    @classmethod
    def _assert_validity(cls):
        # if BaseCB in cls.__bases__:
        assert BaseCB in cls.__mro__ , (cls , cls.__mro__)
        self_hooks , base_hooks = cls._possible_hooks() , BaseCB._possible_hooks()
        invalid_hooks = [x for x in self_hooks if x not in base_hooks]
        if invalid_hooks:
            print(f'Invalid Hooks of {cls.__name__} :' , invalid_hooks)
            print('Use _ or __ to prefix these class-methods')
            raise TypeError(cls)
    @property
    def trace_hook_name(self) -> str:
        env = getattr(currentframe() , 'f_back')
        while not env.f_code.co_name.startswith('on_'): env = getattr(env , 'f_back')
        return env.f_code.co_name
    
class BaseBuffer(ABC):
    '''dynamic buffer space for some module to use (tra), can be updated at each batch / epoch '''
    def __init__(self , device : Optional[Callable] = None , always_on_device = False) -> None:
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
    
    def reset(self): self.contents : dict[str,Any] = {}
    
    @abstractmethod
    def register_setup(self) -> None: ...
    @abstractmethod
    def register_update(self) -> None: ...

class BaseDataModule(ABC):
    '''A class to store relavant training data'''
    @abstractmethod
    def prepare_data() -> None: '''prepare all data in advance of training'''
    @abstractmethod
    def load_data(self) -> None: 
        '''load prepared data at training begin'''
        self.model_date_list : np.ndarray
        self.test_full_dates : np.ndarray
    @abstractmethod
    def setup(self , *args , **kwargs) -> None: 
        '''create train / valid / test dataloaders'''
        self.stage : Literal['fit' , 'test' , 'predict']
        self.y : Tensor
        self.buffer : BaseBuffer
        self.y_secid : Any
        self.y_date : Any
        self.early_test_dates : np.ndarray
        self.model_test_dates : np.ndarray
    @abstractmethod
    def train_dataloader(self) -> Iterator[BatchData | BoosterInput]: '''return train dataloaders'''
    @abstractmethod
    def val_dataloader(self) -> Iterator[BatchData | BoosterInput]: '''return valid dataloaders'''
    @abstractmethod
    def test_dataloader(self) -> Iterator[BatchData | BoosterInput]: '''return test dataloaders'''
    @abstractmethod
    def predict_dataloader(self) -> Iterator[BatchData | BoosterInput]: '''return predict dataloaders'''
    def on_before_batch_transfer(self , batch , dataloader_idx = None): return batch
    def transfer_batch_to_device(self , batch , device = None , dataloader_idx = None): return batch
    def on_after_batch_transfer(self , batch , dataloader_idx = None): return batch
    def reset_dataloaders(self):
        '''reset for every fit / test / predict'''
        self.loader_dict  = {}
        self.loader_param = ()
        self.buffer.reset()
    def prev_model_date(self , model_date):
        prev_dates = [d for d in self.model_date_list if d < model_date]
        return max(prev_dates) if prev_dates else -1
    def next_model_date(self , model_date):
        if model_date < max(self.model_date_list):
            return min(self.model_date_list[self.model_date_list > model_date])
        else:
            return max(self.test_full_dates) + 1

class BaseTrainer(ABC):
    '''run through the whole process of training'''
    @final
    def __init__(self , **kwargs):
        self.init_config(**kwargs)
        self.init_utilities(**kwargs)
        self.init_data(**kwargs)
        self.status = TrainerStatus(self.config['max_epoch'])
        if hasattr(self , 'callbacks'):
            [setattr(self , x , self.callbacks(getattr(self , x))) for x in dir(self) if BaseCB._possible_hook(x)]

    @abstractmethod
    def batch_forward(self) -> None: 
        '''forward of batch_data'''
        self.batch_output = self(self.batch_data)
    @abstractmethod
    def batch_metrics(self) -> None: 
        '''calculate and collect of batch_data'''
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
    @abstractmethod
    def batch_backward(self) -> None: 
        '''backward of batch loss'''
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
    @abstractmethod
    def init_config(self , **kwargs) -> None:
        '''initialized configuration'''
        self.config : Any
        self.stage_queue : list[Literal['data' , 'fit' , 'test']] = ['data' , 'fit' , 'test']
    @abstractmethod
    def init_utilities(self , **kwargs): 
        '''initialized all relevant utilities'''
        self.logger     : Logger
        self.checkpoint : Any
        self.deposition : Any
        self.metrics    : Any
        self.callbacks  : Any
        self.device     : Any
        self.model      : Any
        self.dataloader : Iterable[BatchData | BatchData | Any]
    @abstractmethod
    def init_data(self , **kwargs): 
        '''initialized data_module'''
        self.data : BaseDataModule
    @abstractmethod
    def save_model(self) -> None: 
        '''save self.net to somewhere'''
    @abstractmethod
    def stack_model(self) -> None: 
        '''temporaly save self.net to somewhere'''
    @abstractmethod
    def load_model(self , *args , **kwargs) -> None: 
        '''load self.net to somewhere'''
        self.net : nn.Module
        self.booster : Any
        self.optimizer : Any
    @abstractmethod
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
    @abstractmethod
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
    @abstractmethod
    def model_param(self) -> dict:  '''current model param'''
    @property
    @abstractmethod
    def model_iter(self) -> Iterator[tuple[int,int]]: '''iter of model_date and model_num , considering resume_training'''
    @property
    @abstractmethod
    def if_transfer(self) -> bool: '''whether use last model to refine model'''
    @property
    @abstractmethod
    def model_types(self) -> list[str]: '''iter of model_type'''
    @property
    def batch_dates(self): return np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
    @property
    def batch_warm_up(self): return len(self.data.early_test_dates)
    @property
    def model_date(self): return self.status.model_date
    @property
    def model_num(self): return self.status.model_num
    @property
    def model_type(self): return self.status.model_type
    @property
    def prev_model_date(self): return self.data.prev_model_date(self.model_date)
    
    def __call__(self , input : BatchData | Any):
        if isinstance(input , BatchData):
            if input.is_empty:
                output = None
            else:
                output = self.net(input.x , **input.kwargs)
        else:
            output = self.net(input)
        return BatchOutput(output)

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
            if self.status.fit_iter_num == 0:
                self.logger.warning(f'First Iterance: ({self.status.model_date} , {self.status.model_num})')
            self.fit_model()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.status.stage_test()
        self.on_test_start()
        for self.status.model_date , self.status.model_num in self.model_iter:
            self.test_model()
        self.on_test_end()    

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
    def on_train_epoch_start(self): ...
    def on_train_epoch_end(self): ...
    def on_validation_epoch_start(self): ...
    def on_validation_epoch_end(self): ...
    def on_test_model_start(self): ...
    def on_test_model_end(self): ...
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