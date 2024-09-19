import itertools
import numpy as np

from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from inspect import currentframe

from torch import Tensor
from typing import Any , final , Iterator , Literal , Optional

from .batch import BatchData , BatchOutput
from .buffer import BaseBuffer
from .config import TrainConfig
from .storage import MemFileStorage
from ...basic.util import ModelDict
from ...func import BigTimer , Filtered

class ModelStreamLine(ABC):
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
    def on_test_batch_end(self): ...
    def on_test_batch_start(self): ... 
    def on_test_end(self): ... 
    def on_test_model_end(self): ... 
    def on_test_model_start(self): ... 
    def on_test_submodel_end(self): ... 
    def on_test_submodel_start(self): ... 
    def on_test_start(self): ... 
    def on_train_batch_end(self): ... 
    def on_train_batch_start(self): ... 
    def on_train_epoch_end(self): ... 
    def on_train_epoch_start(self): ... 
    def on_validation_batch_end(self): ... 
    def on_validation_batch_start(self): ...
    def on_validation_epoch_end(self): ...
    def on_validation_epoch_start(self): ...
    
def possible_hooks() -> list[str]: 
    return [x for x in dir(ModelStreamLine) if not x.startswith('_')]

@dataclass
class TrainerStatus:
    max_epoch : int = 200
    stage   : Literal['data' , 'fit' , 'test'] = 'data'
    dataset : Literal['train' , 'valid' , 'test' , 'predict'] = 'train'
    epoch   : int = -1
    attempt : int = 0
    round   : int = 0
    
    model_num  : int = -1
    model_date : int = -1
    model_submodel : str = 'best'
    epoch_event : list[str] = field(default_factory=list)
    best_attempt_metric : Any = None

    def __post_init__(self):
        self.end_of_loop = self.EndofLoop(self.max_epoch)
        self.fit_iter_num : int = 0

    @property
    def status(self):
        return {
            'stage' : self.stage ,
            'dataset' : self.dataset ,
            'model_num' : self.model_num ,
            'model_date' : self.model_date ,
            'model_submodel' : self.model_submodel ,
            'epoch' : self.epoch ,
            'attempt' : self.attempt ,
            'round' : self.round
        }

    def stage_data(self): self.stage = 'data'
    def stage_fit(self):  self.stage = 'fit'
    def stage_test(self): self.stage = 'test'
    def dataset_train(self): self.dataset = 'train'
    def dataset_validation(self): self.dataset = 'valid'
    def dataset_test(self): self.dataset = 'test'
    def fit_model_start(self):
        self.fit_iter_num += 1
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
        
class BaseDataModule(ABC):
    '''A class to store relavant training data'''
    @abstractmethod
    def __init__(self , config : Optional[TrainConfig] = None , use_data : Literal['fit','predict','both'] = 'fit'):
        self.config   : TrainConfig
        self.use_data : Literal['fit','predict','both'] 
        self.storage  : MemFileStorage
        self.buffer   : BaseBuffer
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
        self.y_secid : Any
        self.y_date : Any
        self.early_test_dates : np.ndarray
        self.model_test_dates : np.ndarray
    @abstractmethod
    def train_dataloader(self)  -> Iterator[BatchData]: '''return train dataloaders'''
    @abstractmethod
    def val_dataloader(self)    -> Iterator[BatchData]: '''return valid dataloaders'''
    @abstractmethod
    def test_dataloader(self)   -> Iterator[BatchData]: '''return test dataloaders'''
    @abstractmethod
    def predict_dataloader(self)-> Iterator[BatchData]: '''return predict dataloaders'''
    def on_before_batch_transfer(self , batch , dataloader_idx = None): return batch
    def transfer_batch_to_device(self , batch , device = None , dataloader_idx = None): return batch
    def on_after_batch_transfer(self , batch , dataloader_idx = None): return batch
    def reset_dataloaders(self):
        '''reset for every fit / test / predict'''
        self.loader_dict  = {}
        self.loader_param = ()
    def prev_model_date(self , model_date):
        prev_dates = [d for d in self.model_date_list if d < model_date]
        return max(prev_dates) if prev_dates else -1
    def next_model_date(self , model_date):
        if model_date < max(self.model_date_list):
            return min(self.model_date_list[self.model_date_list > model_date])
        else:
            return max(self.test_full_dates) + 1
        
    @property
    def device(self): return self.config.device

class BaseTrainer(ModelStreamLine):
    '''run through the whole process of training'''
    def __bool__(self): return True
    @final
    def __init__(self , base_path = None , **kwargs):
        self.init_config(base_path = base_path ,**kwargs)
        self.init_data(**kwargs)
        self.init_model(**kwargs)
        self.init_callbacks(**kwargs)
        self.wrap_callbacks()
        
    def init_config(self , base_path = None , **kwargs) -> None:
        '''initialized configuration'''
        self.config = TrainConfig.load(base_path , do_parser = True , par_args = kwargs)
        self.status = TrainerStatus(self.config.train_max_epoch)

    def wrap_callbacks(self):
        [setattr(self , hook_name , self.hook_wrapper(self , self.model , self.callback , hook_name)) 
         for hook_name in possible_hooks()]

    @staticmethod
    def hook_wrapper(trainer : 'BaseTrainer' , model : 'BasePredictorModel' , callback : 'BaseCallBack' , hook_name : str):
        action_trainer = getattr(trainer , hook_name)
        action_model = getattr(model , hook_name)
        def wrapper() -> None:
            callback.at_enter(hook_name)
            action_trainer()
            action_model()
            callback.at_exit(hook_name)
        return wrapper

    @abstractmethod
    def init_model(self , **kwargs): 
        '''initialized data_module'''
        self.model  : BasePredictorModel

    @abstractmethod
    def init_callbacks(self , **kwargs): 
        '''initialized data_module'''
        self.callback  : BaseCallBack

    @abstractmethod
    def init_data(self , **kwargs): 
        '''initialized data_module'''
        self.data : BaseDataModule

    @property
    def device(self): return self.config.device
    @property
    def metrics(self):  return self.config.metrics
    @property
    def checkpoint(self): return self.config.checkpoint
    @property
    def deposition(self): return self.config.deposition
    @property
    def logger(self): return self.config.logger    
    @property
    def stage_queue(self): return self.config.stage_queue
    @property
    def batch_dates(self): return np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
    @property
    def batch_warm_up(self): return len(self.data.early_test_dates)
    @property
    def batch_aftermath(self): return len(self.data.early_test_dates) + len(self.data.model_test_dates)
    @property
    def model_date(self): return self.status.model_date
    @property
    def model_num(self): return self.status.model_num
    @property
    def model_submodel(self): return self.status.model_submodel
    @property
    def prev_model_date(self): return self.data.prev_model_date(self.model_date)
    @property
    def model_param(self): return self.config.Model.params[self.model_num]
    @property
    def model_submodels(self): return self.config.model_submodels
    @property
    def if_transfer(self): return self.config.train_trainer_transfer     
    
    @property
    def batch_output(self): return self.model.batch_output
    
    def main_process(self):
        '''Main stage of data & fit & test'''
        self.on_configure_model()
        for self.stage in self.stage_queue: 
            getattr(self , f'stage_{self.stage}')()
        self.on_summarize_model()

    def go(self):
        with BigTimer(self.logger.critical , 'Main Process'):
            self.main_process()
        return self

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
        for self.status.model_date , self.status.model_num in self.iter_model_num_date():
            if self.status.fit_iter_num == 0:
                self.logger.warning(f'First Iterance: ({self.status.model_date} , {self.status.model_num})')
            self.on_fit_model_start()
            self.model.fit()
            self.on_fit_model_end()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.status.stage_test()
        self.on_test_start()
        for self.status.model_date , self.status.model_num in self.iter_model_num_date():
            self.on_test_model_start()
            self.model.test()
            self.on_test_model_end()
        self.on_test_end()    

    def iter_model_num_date(self): 
        '''iter of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(self.data.model_date_list , self.config.model_num_list))
        if self.config.resume_training and self.status.stage == 'fit':
            models_trained = np.full(len(new_iter) , True , dtype = bool)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not self.deposition.exists(model_num , model_date):
                    models_trained[max(i,0):] = False
                    break
            new_iter = Filtered(new_iter , ~models_trained)
        return new_iter

    def iter_model_submodels(self):
        for self.status.model_submodel in self.model_submodels: 
            self.on_test_submodel_start()
            yield self.status
            self.on_test_submodel_end()

    def iter_fit_epoches(self):
        while not self.status.end_of_loop:
            self.on_fit_epoch_start()
            yield self.status
            self.on_before_fit_epoch_end()
            self.on_fit_epoch_end()

    def iter_train_dataloader(self , given_loader = None):
        self.dataloader = self.data.train_dataloader() if given_loader is None else given_loader
        self.on_train_epoch_start()
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_train_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_train_batch_end()
        self.on_train_epoch_end()

    def iter_val_dataloader(self , given_loader = None):
        self.dataloader = self.data.val_dataloader() if given_loader is None else given_loader
        self.on_validation_epoch_start()
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_validation_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_validation_batch_end()
        self.on_validation_epoch_end()

    def iter_test_dataloader(self , given_loader = None):
        self.dataloader = self.data.test_dataloader() if given_loader is None else given_loader
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_test_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_test_batch_end()

    def iter_predict_dataloader(self , given_loader = None):
        self.dataloader = self.data.predict_dataloader() if given_loader is None else given_loader
        for self.batch_idx , self.batch_data in enumerate(self.dataloader): 
            self.on_test_batch_start()
            yield self.batch_idx , self.batch_data
            self.on_test_batch_end()

    def stack_model(self):
        '''temporaly save self to somewhere'''
        self.on_before_save_model()
        for submodel in self.model_submodels:
            model_dict = self.model.collect(submodel)
            self.deposition.stack_model(model_dict , self.model_num , self.model_date , submodel) 

    def save_model(self):
        '''save self to somewhere'''
        if self.metrics.better_attempt(self.status.best_attempt_metric): self.stack_model()
        [self.deposition.dump_model(self.model_num , self.model_date , submodel) for submodel in self.model_submodels]

    def on_configure_model(self):  
        self.config.set_config_environment()

    def on_fit_model_start(self):
        self.status.fit_model_start()
        self.data.setup('fit' , self.model_param , self.model_date)

    def on_fit_model_end(self): 
        self.save_model()

    def on_fit_epoch_start(self):
        self.status.fit_epoch_start()

    def on_fit_epoch_end(self):
        self.status.fit_epoch_end()

    def on_train_epoch_start(self): 
        self.status.dataset_train()
        self.metrics.new_epoch(**self.status.status)

    def on_train_epoch_end(self):
        self.metrics.collect_epoch()
    
    def on_validation_epoch_start(self):
        self.status.dataset_validation()
        self.metrics.new_epoch(**self.status.status)

    def on_validation_epoch_end(self):
        self.metrics.collect_epoch()
    
    def on_test_model_start(self):
        self.status.dataset_test()
        self.data.setup('test' , self.model_param , self.model_date)
    
    def on_test_submodel_start(self):
        self.metrics.new_epoch(**self.status.status)
        assert self.deposition.exists(self.model_num , self.model_date , self.model_submodel) , \
            (self.model_num , self.model_date , self.model_submodel)
        
    def on_test_submodel_end(self): 
        self.metrics.collect_epoch()

    def on_test_batch_start(self):
        self.assert_equity(self.batch_dates[self.batch_idx] , self.data.y_date[self.batch_data.i[0,1]]) 

    @property
    def penalty_kwargs(self): return {}
    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)

class ModelStreamLineWithTrainer(ModelStreamLine):
    def bound_with_trainer(self , trainer): 
        self.trainer : BaseTrainer | Any = trainer
        return self

    @property
    def config(self): return self.trainer.config
    @property
    def status(self):  return self.trainer.status
    @property
    def logger(self): return self.config.logger
    @property
    def metrics(self):  return self.config.metrics
    @property
    def checkpoint(self): return self.config.checkpoint
    @property
    def deposition(self): return self.config.deposition
    @property
    def device(self): return self.config.device
    @property
    def data(self): return self.trainer.data
    @property
    def batch_data(self): return self.trainer.batch_data
    @property
    def batch_idx(self): return self.trainer.batch_idx
    @property
    def verbosity(self): return self.config.verbosity
    @property
    def model_date(self): return self.trainer.model_date
    @property
    def model_num(self): return self.trainer.model_num
    @property
    def model_submodel(self): return self.trainer.model_submodel

class BaseCallBack(ModelStreamLineWithTrainer):
    def __init__(self , trainer , turn_off = False) -> None:
        self.bound_with_trainer(trainer)
        self.turn_off : bool = turn_off
        self.__hook_stack = []

    def print_info(self , depth = 0 , **kwargs):
        frame = currentframe()
        for _ in range(depth + 1): frame = getattr(frame , 'f_back')
        args = {k:v for k,v in getattr(frame , 'f_locals').items() if k not in ['self','trainer','kwargs'] and not k.startswith('_')}
        args.update(kwargs)
        info = self.__class__.__name__ + '({})'.format(','.join([f'{k}={v}' for k,v in args.items()])) 
        if self.__class__.__doc__: info += f' , {self.__class__.__doc__}'
        print(info)

    def __enter__(self): 
        self.__hook_stack.append(self.trace_hook_name())
        self.at_enter(self.__hook_stack[-1])
    def __exit__(self , *args): self.at_exit(self.__hook_stack.pop())
    def at_enter(self , hook_name : str):  ...
    def at_exit(self , hook_name : str): getattr(self , hook_name)()

    def trace_hook_name(self) -> str:
        env = getattr(currentframe() , 'f_back')
        while not env.f_code.co_name.startswith('on_'): env = getattr(env , 'f_back')
        return env.f_code.co_name
    
    @classmethod
    def possible_hooks(cls): return possible_hooks()

    @property
    def model(self): return self.trainer.model

class BasePredictorModel(ModelStreamLineWithTrainer):
    '''a group of ensemble models , of same net structure'''
    AVAILABLE_CALLBACKS = []
    COMPULSARY_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']
    
    def __init__(self, *args , **kwargs) -> None:
        self.reset()
        self.model_dict = ModelDict()

    def __call__(self , input : BatchData | Tensor | Any , *args , **kwargs):
        if input is None or len(input) == 0:
            output = None
        else:
            output = self.forward(input , *args , **kwargs)
        batch_output = BatchOutput(output)
        return batch_output
    
    def multiloss_params(self): return {}

    def reset(self):
        self.trainer : BaseTrainer | Any = None
        self._config : TrainConfig | Any = None
        return self

    def bound_with_config(self , config : TrainConfig):
        assert self.trainer is None , 'Cannot bound with config if bound with trainer first'
        self._config = config
        return self.init_utils()

    def bound_with_trainer(self , trainer : BaseTrainer):
        self.reset()
        self.trainer = trainer
        return self.init_utils()
    
    def init_utils(self):
        self.config.init_utils()
        return self

    @classmethod
    def create_from_trainer(cls , trainer : BaseTrainer):
        return cls().bound_with_trainer(trainer)

    @property
    def config(self):
        return self.trainer.config if self.trainer else self._config
    @property
    def model_num(self):
        return self.trainer.model_num if self.trainer else self._model_num
    @property
    def model_date(self):
        return self.trainer.model_date if self.trainer else self._model_date
    @property
    def model_submodel(self):
        return self.trainer.model_submodel if self.trainer else self._model_submodel
    @property
    def model_param(self): return self.config.model_param[self.model_num]
    
    def load_model_file(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when fitting/testing new model'''
        if model_num is not None: self._model_num  = model_num
        else: model_num = self.model_num
        if model_date is not None: self._model_date = model_date
        else: model_date = self.model_date
        if submodel is not None: self._model_submodel = submodel
        else: submodel = self.model_submodel
        return self.deposition.load_model(model_num , model_date , submodel)
    
    @abstractmethod
    def new_model(self , *args , **kwargs):
        '''call when fitting new model'''
        self.optimizer : Any
        return self
    @abstractmethod
    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when testing new model'''
        return self
    @abstractmethod
    def forward(self , batch_data : BatchData | Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
    @abstractmethod
    def fit(self) -> None:
        '''fit the model inside'''
    @abstractmethod
    def collect(self , submodel = 'best' , *args) -> ModelDict: 
        '''collect model params, called before stacking model'''

    def test(self):
        '''test the model inside'''
        for _ in self.trainer.iter_model_submodels():
            self.load_model(submodel=self.model_submodel)
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

    def metric_kwargs(self):
        pred   = self.batch_output.pred
        label  = self.batch_data.y
        weight = self.batch_data.w
        multiloss = self.multiloss_params()
        return {'pred':pred,'label':label,'weight':weight,'multiloss':multiloss,**self.batch_output.other}
    
    def batch_forward(self) -> None: 
        if self.status.dataset == 'test':
            if self.trainer.batch_idx >= self.trainer.batch_aftermath: return
        self.batch_output = self(self.batch_data)

    def batch_metrics(self) -> None:
        if self.batch_data.is_empty: return
        if self.status.dataset == 'test':
            if self.trainer.batch_idx < self.trainer.batch_warm_up: return
            if self.trainer.batch_idx >= self.trainer.batch_aftermath: return
        '''if net has multiloss_params , get it and pass to calculate_from_tensor'''
        self.metrics.calculate(self.status.dataset , **self.metric_kwargs()).collect_batch()

    def batch_backward(self) -> None:
        if self.batch_data.is_empty: return
        assert self.status.dataset == 'train' , self.status.dataset
        self.trainer.on_before_backward()
        self.optimizer.backward(self.metrics.output)
        self.trainer.on_after_backward()

