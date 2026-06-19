"""
Base trainer class for the project
"""
from __future__ import annotations

import itertools
import numpy as np

from functools import wraps
from typing import Any , Literal , TypeAlias , cast , Sized , Callable

from src.proj import Const , Base

from src.proj.bases import FittingEventType
from src.res.model.util.core import BatchOutput , BatchData , epoch_key
from src.res.model.util.config import ModelConfig
from .pipeline import BasePipeline
from .future_utils import FutureUtils

__all__ = ['BaseTrainer']

NewAttemptType : TypeAlias = Literal['model' , 'attempt']

class TrainerHookWrapper:
    """
    Wrapper for hooks, used to wrap the hooks of the trainer and the model
    The order of the hooks is:
    - callback enter hook
    - status hooks
    - trainer hooks (self)
    - model hooks
    - metrics hooks
    - predrecorder hooks
    - callback exit hooks
    """
    wrapped_records : dict[int , bool] = {}

    @classmethod
    def wrap(cls , trainer : BaseTrainer):
        key = id(trainer)
        assert not cls.wrapped_records.get(key , False) , f'Hooks already wrapped for {trainer.__class__.__name__}'
        for hook in trainer.all_hooks:
            func = getattr(trainer , hook)
            setattr(trainer , f'_raw_{hook}' , func)
            setattr(trainer , hook , cls.wrap_single_hook(trainer , hook , func))
        cls.wrapped_records[key] = True

    @classmethod
    def wrap_single_hook(cls , trainer : BaseTrainer , hook : str , func : Callable):
        @wraps(func)
        def wrapper(*args , **kwargs) -> None:
            trainer.callback.at_enter(hook , *args , **kwargs)
            trainer.status.execute_hook(hook , *args , **kwargs)
            func(*args , **kwargs)
            trainer.model.execute_hook(hook , *args , **kwargs)
            trainer.metrics.execute_hook(hook , *args , **kwargs)
            trainer.record.execute_hook(hook , *args , **kwargs)
            trainer.callback.at_exit(hook , *args , **kwargs)
        return wrapper

class BaseTrainer(BasePipeline):
    """run through the whole process of training"""
    _trainer : BaseTrainer | None = None

    def __new__(cls , *args , **kwargs):
        if cls._trainer is None:
            obj = super().__new__(cls)
            TrainerHookWrapper.wrap(obj)
            cls._trainer = obj
        return cls._trainer
    
    def __init__(self , base_path = None , * , 
                 module : str | None = None , schedule_name = None , 
                 override : dict | None = None , 
                 use_data : Base.lit.DataBlockTimeFrames = 'fit' , 
                 indent : int = 0 , vb_level : Base.lit.VerbosityLevel = 1 , **kwargs):
        assert use_data != 'predict' , 'use_data cannot be predict when training models'
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self._config_kwargs = {
            'base_path': base_path,
            'module': module,
            'schedule_name': schedule_name,
            'override': override,
            'indent': indent,
            'vb_level': vb_level,
        }
        self._use_data = use_data
        self._kwargs = kwargs

    def __bool__(self): 
        return True

    def __repr__(self): 
        return f'{self.__class__.__name__}(path={self.base_path})'

    def init_config(self) -> None:
        self._config   = ModelConfig.initialize(**self._config_kwargs , **self._kwargs)

    def init_cores(self) -> None:
        """
        init core components of the trainer: config, data, model, callbacks
        """
        self._model    = FutureUtils.model(self)
        self._callback = FutureUtils.callback(self)
        self._data     = FutureUtils.data(self)

    def init_utils(self):
        """
        init utils of the trainer: status, record, texts, container, metrics, checkpoint, deposition
        """
        self._status     = FutureUtils.status(self)
        self._checkpoint = FutureUtils.checkpoint(self)
        self._container  = FutureUtils.container(self)
        self._texts      = FutureUtils.texts(self)
        self._record     = FutureUtils.record(self)
        self._metrics    = FutureUtils.metrics(self)
        self._deposition = FutureUtils.deposition(self)

    @property
    def base_path(self):
        return self.config.base_path if hasattr(self , '_config') else self._config_kwargs['base_path']
    @property
    def use_data(self) -> Base.lit.DataBlockTimeFrames:
        return cast(Base.lit.DataBlockTimeFrames, self._use_data)
    @property
    def input_model_kwargs(self):
        return self._kwargs
    @property
    def config(self):
        """config of the trainer , class of ModelConfig"""
        return self._config
    @property
    def data(self):
        """data of the trainer , class of DataModule"""
        return self._data
    @property
    def model(self):
        """model of the trainer , class of BasePredictorModel"""
        return self._model
    @property
    def callback(self):
        """callback of the trainer , class of CallBackManager, include all callbacks"""
        return self._callback
    @property
    def status(self): 
        """status of the trainer , class of TrainerStatus"""
        return self._status
    @property
    def record(self): 
        """record of the trainer , class of PredRecorder"""
        return self._record
    @property
    def texts(self): 
        """texts of the trainer , class of TrainerTexts"""
        return self._texts
    @property
    def container(self): 
        """container of the trainer , class of TrainerContainer"""
        return self._container
    @property
    def metrics(self): 
        """metrics of the trainer , class of Metrics"""
        return self._metrics
    @property
    def checkpoint(self): 
        """checkpoint of the trainer , class of Checkpoint"""
        return self._checkpoint
    @property
    def deposition(self): 
        """deposition of the trainer , class of Deposition"""
        return self._deposition
    @property
    def device(self): 
        return self.config.device
    @property
    def queue_of_stages(self): 
        return self.config.queue_of_stages
    @property
    def batch_num(self): 
        assert isinstance(self.dataloader , Sized) , f'dataloader is not a Sized object: {self.dataloader}'
        return len(self.dataloader)
    @property
    def batch_data(self): 
        return BatchData(self.batch_input , self.batch_output)
    @property
    def batch_dates(self) -> np.ndarray: 
        return self.cached_properties.query('model_start' , 'batch_dates' , lambda: np.concatenate([self.data.early_test_dates , self.data.model_test_dates]))
    @property
    def batch_warm_up(self) -> int: 
        return self.cached_properties.query('model_start' , 'batch_warm_up' , lambda: len(self.data.early_test_dates) if self.status.stage == 'test' else 0)
    @property
    def batch_aftermath(self) -> int: 
        return self.cached_properties.query('model_start' , 'batch_aftermath' , lambda: len(self.data.early_test_dates) + len(self.data.model_test_dates) if self.status.stage == 'test' else 100_000_000)
    @property
    def batch_resumed(self) -> int: 
        if not self.cached_properties.has('model_start' , 'batch_resumed'):
            if (
                self.status.stage == 'test' and 
                self.batch_warm_up == 0 and 
                self.config.is_resuming and  
                Const.Model.resume_test and
                Const.Model.resume_test_start == 'last_pred_date'):
                value = sum(self.data.model_test_dates <= self.config.resumed_max_pred_date)
            else:
                value = 0
            self.cached_properties.set('model_start' , 'batch_resumed' , value)
        return self.cached_properties.get('model_start' , 'batch_resumed')

    @property
    def model_date(self): 
        return self.status.model_date
    @property
    def model_num(self): 
        return self.status.model_num
    @property
    def model_submodel(self): 
        return self.status.model_submodel

    @property
    def prev_model_date(self): 
        return self.data.prev_model_date(self.model_date)
    @property
    def model_param(self): 
        return self.config.algo_config.params[self.model_num]
    @property
    def model_submodels(self): 
        return self.config.submodels
    @property
    def if_transfer(self): 
        return self.config.transfer_training     
    @property
    def html_catcher_export_path(self): 
        if 'fit' in self.queue_of_stages and 'test' in self.queue_of_stages:
            status = 'fitting_testing'
        elif 'test' in self.queue_of_stages:
            status = 'testing'
        elif 'fit' in self.queue_of_stages:
            status = 'fitting'
        else:
            status = 'unknown'
        return self.config.base_path.rslt(f'{status}_output.html')
    @property
    def is_fitting(self): 
        return self.status.stage == 'fit'
    
    def main_process(self):
        """Main stage of data & fit & test"""
        self.stage_setup()

        if 'data' in self.queue_of_stages:
            self.stage_data()

        if 'fit' in self.queue_of_stages:  
            self.stage_fit()

        if 'test' in self.queue_of_stages: 
            self.stage_test()
        
        self.stage_summary()

        return self

    def go(self):
        """alias of main_process"""
        return self.main_process()

    def stage_setup(self):
        """stage of setting up"""
        with self.logger.paragraph('Stage [Setup]' , 2):
            self.init_config()
            self.init_cores()
            self.init_utils()
            self.on_configure_model()
            self.print_out()

    def stage_data(self):
        """stage of loading model data"""
        with self.logger.paragraph('Stage [Data]' , 2):
            self.on_data_start_before()
            self.on_data_start()
            self.data.load_data()
            self.on_data_end()
            self.on_data_end_after()
        
    def stage_fit(self):
        """stage of fitting"""
        with self.logger.paragraph('Stage [Fit]' , 2):
            self.config.log_operation('fit' , 'start')
            self.on_fit_start_before()
            self.on_fit_start()
            for self.status.model_date , self.status.model_num in self.iter_model_num_date():
                if self.status.model_num == 0:
                    self.on_fit_model_date_start()
                self.on_fit_model_start()
                self.model.fit()
                self.on_fit_model_end()
                if self.status.model_num == self.config.model_num:
                    self.on_fit_model_date_end()
            self.on_fit_end()
            self.on_fit_end_after()
            self.config.log_operation('fit' , 'end')

    def stage_test(self):
        """stage of testing"""
        with self.logger.paragraph('Stage [Test]' , 2):
            self.config.log_operation('test' , 'start')
            self.on_test_start_before()
            self.on_test_start()
            for self.status.model_date , self.status.model_num in self.iter_model_num_date():
                if self.status.model_num == 0:
                    self.on_test_model_date_start()
                self.on_test_model_start()
                self.model.test()
                self.on_test_model_end()
                if self.status.model_num == self.config.model_num_list[-1]:
                    self.on_test_model_date_end()
            self.on_test_end_before()
            self.on_test_end()
            self.on_test_end_after()
            self.config.log_operation('test' , 'end')

    def stage_summary(self):
        """stage of summarizing"""
        with self.logger.paragraph('Stage [Summary]' , 2):
            self.on_summarize_model()

    def iter_model_num_date(self): 
        """iter of model_date and model_num , considering is_resuming"""
       
        model_iter = list(itertools.product(self.data.model_date_list , self.config.model_num_list))
        assert self.status.stage in ['fit' , 'test'] , self.status.stage
        num_all_models = len(model_iter)
        iter_info = f'In Stage [{self.status.stage}], number of all models (model_date x model_num) is {num_all_models}, '
        if self.config.is_resuming:
            match self.status.stage:
                case 'fit':
                    models_trained = np.full(len(model_iter) , True , dtype = bool)
                    for i , (model_date , model_num) in enumerate(model_iter):
                        if not self.deposition.exists(model_num , model_date):
                            models_trained[max(i,0):] = False
                            break
                    condition = ~models_trained
                case 'test':
                    resumed_models_finished = self.record.resumed_models.groupby(['model_date' , 'model_num']).groups
                    resumed = [(model_date , model_num) not in resumed_models_finished for model_date , model_num in model_iter]
                    condition = np.array(resumed)
                case _:
                    self.logger.error(f'Invalid stage for resuming iter_model_num_date: {self.status.stage}')
                    condition = np.full(len(model_iter) , True , dtype = bool)
            model_iter = Base.FilteredIterable(model_iter , condition)
            iter_info += f'resuming {num_all_models - sum(condition)} models, {sum(condition)} to go!'
        else:
            iter_info += f'{num_all_models} to go!'
        self.logger.note(iter_info , vb = 1)
        return model_iter

    def iter_model_submodels_on_save(self):
        assert self.is_fitting , f'{self.status.stage} is not allowed to iter model submodels on save'
        for self.status.model_submodel in self.model_submodels: 
            yield self.status
            
    def iter_model_submodels(self):
        assert not self.is_fitting , f'{self.status.stage} is not allowed to iter model submodels'
        for self.status.model_submodel in self.model_submodels: 
            self.on_test_submodel_start()
            yield self.status
            self.on_test_submodel_end()
        
    def iter_fit_epoches(self):
        while not self.status.loop_end:
            self.on_fit_epoch_start()
            yield self.status
            self.on_fit_epoch_end_before()
            self.on_fit_epoch_end()

    def iter_dataloader(self):
        for idx , batch_input in enumerate(self.dataloader):
            self.batch_idx = idx
            self.batch_input = batch_input
            self.batch_output = BatchOutput()
            yield None

    def iter_train_dataloader(self , given_loader = None):
        self.dataloader = self.data.train_dataloader() if given_loader is None else given_loader
        self.on_train_epoch_start()
        for _ in self.iter_dataloader(): 
            self.on_train_batch_start()
            yield self.batch_idx , self.batch_input
            self.on_train_batch_end()
        self.on_train_epoch_end()

    def iter_val_dataloader(self , given_loader = None):
        self.dataloader = self.data.val_dataloader() if given_loader is None else given_loader
        self.on_validation_epoch_start()
        for _ in self.iter_dataloader(): 
            self.on_validation_batch_start()
            yield self.batch_idx , self.batch_input
            self.on_validation_batch_end()
        self.on_validation_epoch_end()

    def iter_test_dataloader(self , given_loader = None):
        self.dataloader = self.data.test_dataloader() if given_loader is None else given_loader
        for _ in self.iter_dataloader(): 
            self.on_test_batch_start()
            yield self.batch_idx , self.batch_input
            self.on_test_batch_end()

    def iter_predict_dataloader(self , given_loader = None):
        self.dataloader = self.data.predict_dataloader() if given_loader is None else given_loader
        for _ in self.iter_dataloader(): 
            self.on_test_batch_start()
            yield self.batch_idx , self.batch_input
            self.on_test_batch_end()

    def on_configure_model(self):  
        self.config.set_config_environment()
        
    def on_fit_model_start(self):
        self.cached_properties.clear('model_start')
        self.data.setup('fit' , self.model_param , self.model_date)
        self.new_attempt('model')

    def on_fit_epoch_end_before(self):
        self.status.check_loop_end(self.config.max_epoch)

    def on_fit_model_end(self): 
        self.model.stack_model()
        self.model.dump_model()
        
    def on_test_model_start(self):
        self.cached_properties.clear('model_start')
        self.data.setup('test' , self.model_param , self.model_date)
        
    def on_test_submodel_start(self):
        try:
            self.model.load_model(submodel=self.model_submodel)
        except FileNotFoundError:
            self.status.deposition_loading_status = False

    def on_test_batch_start(self):
        self.assert_date_equity()

    def assert_date_equity(self): 
        date0 = self.batch_dates[self.batch_idx] 
        date1 = self.batch_input.date0 
        if not date0 == date1:
            self.logger.alert1(f'y_date: {self.data.y_date}')
            self.logger.alert1(f'batch_idx: {self.batch_idx}')
            self.logger.alert1(f'batch_dates: {self.batch_dates}')
            self.logger.alert1(f'early_test_dates: {self.data.early_test_dates}')
            self.logger.alert1(f'model_test_dates: {self.data.model_test_dates}')
            self.logger.alert1(f'batch_input.i: {self.batch_input.i[0]}')
            self.logger.alert1(f'batch_input.date0: {self.batch_input.date0}')
            self.logger.alert1(f'Date equity assertion failed: {date0} != {date1}')
            raise ValueError(f'Date equity assertion failed: {date0} != {date1}')

    def print_out(self):
        self.config.print_out(vb_level = 2 , min_key_len = 30)
        self.data.print_out(vb_level = 2 , min_key_len = 30)
        self.model.print_out(vb_level = 2 , min_key_len = 30)
        self.callback.print_out(vb_level = 2 , min_key_len = 30)

    def new_attempt(self , type : NewAttemptType , **kwargs):
        self.model.new_model(**kwargs)
        self.checkpoint.new_model(**self.status.status)
        if type == 'attempt':
            self.metrics.new_attempt()
        elif type == 'model':
            self.metrics.new_model(self.model , self.model.complete_model_param)
        self.on_new_attempt()
        
    def recall_ckpt(self , epoch : int , phase : int = 0 , message : str = '' , details : dict[str, Any] | None = None):
        assert epoch >= 0 and epoch <= self.status.epoch , f'epoch {epoch} is out of range(0,{self.status.epoch})'
        self.status.add_epoch_event(FittingEventType.NEW_PHASE_RECALL , f'Recall {epoch_key(epoch , phase)}' , epoch , message , details)
        self.status.set_milestone_epoch(epoch + 1)
        if epoch != self.status.epoch:
            self.model.load_state_dict(self.checkpoint.load(epoch , phase))
        self.on_new_phase()