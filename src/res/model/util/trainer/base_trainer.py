from __future__ import annotations

import itertools
import numpy as np

from abc import abstractmethod
from typing import Any , final , Literal , Sized
from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter

from src.proj import Proj , Logger , PATH , Const
from src.proj.util import FilteredIterable
from src.res.algo import AlgoModule

from src.res.model.util.core import BatchOutput , BatchData , ModelConfig
from .streamline import ModelStreamLine

__all__ = ['BaseTrainer' , 'ModelStreamLineWithTrainer']

class TrainerHookWrapper:
    """
    Wrapper for hooks, used to wrap the hooks of the trainer and the model
    The order of the hooks is:
    - callback enter hook
    - status hooks
    - trainer hooks
    - model hooks
    - predrecorder hooks
    - callback exit hooks
    """
    wrapped_records : dict[int , bool] = {}
    max_wrap_count = 1

    @classmethod
    def wrap(cls , trainer : BaseTrainer):
        key = id(trainer)
        assert not cls.wrapped_records.get(key , False) , f'Hooks already wrapped for {trainer.__class__.__name__}'
        for hook in trainer.base_hooks():
            setattr(trainer , f'_raw_{hook}' , getattr(trainer , hook))
            setattr(trainer , hook , cls.wrap_single_hook(trainer , hook))
        cls.wrapped_records[key] = True

    @classmethod
    def wrap_single_hook(cls , trainer : BaseTrainer , hook : str):
        def wrapper(*args , **kwargs) -> None:
            Logger.stdout(f'{hook} of stage {trainer.status.stage} start' , vb_level = Proj.vb.get('callback'))
            trainer.callback.at_enter(hook , Proj.vb.get('callback'))
            trainer.status.execute_hook(hook)
            getattr(trainer , f'_raw_{hook}')(*args , **kwargs)
            trainer.model.execute_hook(hook)
            trainer.record.execute_hook(hook)
            trainer.callback.at_exit(hook , Proj.vb.get('callback'))
            Logger.stdout(f'{hook} of stage {trainer.status.stage} end' , vb_level = Proj.vb.get('callback'))
        return wrapper

class BaseTrainer(ModelStreamLine):
    '''run through the whole process of training'''
    _trainer : BaseTrainer | None = None

    def __new__(cls , *args , **kwargs):
        if cls._trainer is None:
            obj = super().__new__(cls)
            TrainerHookWrapper.wrap(obj)
            cls._trainer = obj
        return cls._trainer
    
    @final
    def __init__(self , base_path = None , * , 
                 module : str | None = None , schedule_name = None , 
                 override : dict | None = None , 
                 use_data : Literal['fit','predict','both'] = 'fit' , **kwargs):
        with Logger.Paragraph('Stage [Setup]' , 2):
            self.init_config(base_path = base_path , module = module , schedule_name = schedule_name , override = override , **kwargs)
            self.init_data(use_data = use_data , **kwargs)
            self.init_model(**kwargs)
            self.init_callbacks(**kwargs)
            self.init_utils(**kwargs)

    def __bool__(self): return True

    def __repr__(self): 
        return f'{self.__class__.__name__}(path={self.config.base_path.base})'
        
    @abstractmethod
    def init_config(self , base_path = None , * , module : str | None = None , schedule_name = None , override : dict | None = None , **kwargs) -> None:
        '''init configuration'''
        self.config : ModelConfig
    @abstractmethod
    def init_model(self , **kwargs): 
        '''init data_module'''
        from src.res.model.util.trainer import BasePredictorModel
        self.model : BasePredictorModel
    @abstractmethod
    def init_callbacks(self , **kwargs): 
        '''init callbacks'''
        from src.res.model.util.trainer import BaseCallBack
        self.callback : BaseCallBack
    @abstractmethod
    def init_data(self , use_data : Literal['fit','predict','both'] = 'fit' , **kwargs): 
        '''init data_module'''
        from src.res.model.util.data import BaseDataModule
        self.data : BaseDataModule

    def init_utils(self , **kwargs):
        from src.res.model.util.trainer import TrainerStatus , PredRecorder , Metrics
        from src.res.model.util.core import Checkpoint , Deposition
        self.status = TrainerStatus(self.config.max_epoch)
        self.record = PredRecorder(self)
        self.metrics = Metrics.from_config(self.config)
        
        self.checkpoint = Checkpoint(self.config.mem_storage)
        self.deposition = Deposition(self.config.base_path)

    @property
    def device(self): return self.config.device
    @property
    def base_path(self): return self.config.base_path
    @property
    def queue_of_stages(self): return self.config.queue_of_stages
    @property
    def batch_num(self): 
        assert isinstance(self.dataloader , Sized) , f'dataloader is not a Sized object: {self.dataloader}'
        return len(self.dataloader)
    @property
    def batch_data(self): 
        return BatchData(self.batch_input , self.batch_output)
    @property
    def batch_dates(self): 
        return np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
    @property
    def batch_warm_up(self): 
        return len(self.data.early_test_dates) if self.status.stage == 'test' else 0
    @property
    def batch_aftermath(self): 
        return len(self.data.early_test_dates) + len(self.data.model_test_dates) if self.status.stage == 'test' else np.inf
    @property
    def batch_resumed(self): 
        if (
            self.status.stage == 'test' and 
            self.batch_warm_up == 0 and 
            self.config.is_resuming and  
            Const.Model.resume_test and
            Const.Model.resume_test_start == 'last_pred_date'):
            return sum(self.data.model_test_dates <= self.record.resumed_max_pred_date)
        else:
            return 0
    @property
    def model_date(self): return self.status.model_date
    @property
    def model_num(self): return self.status.model_num
    @property
    def model_submodel(self): return self.status.model_submodel
    @property
    def model_str(self): return f'{self.config.model_name}.{self.model_num}.{self.model_submodel}.{self.model_date}'
    @property
    def prev_model_date(self): return self.data.prev_model_date(self.model_date)
    @property
    def model_param(self): return self.config.algo_config.params[self.model_num]
    @property
    def model_submodels(self): return self.config.submodels
    @property
    def if_transfer(self): return self.config.transfer_training     
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
    def model_tensorboad_dir(self):
        return self.config.base_path.snapshot('tensorboard' , f'{self.config.base_path.model_clean_name}.{self.model_num}.{self.model_date}')

    def main_process(self):
        '''Main stage of data & fit & test'''
        self.on_configure_model()

        if not self.queue_of_stages:
            Logger.error("stage_queue is empty , please check src.proj.Proj.States.trainer")
            raise Exception("stage_queue is empty , please check src.proj.Proj.States.trainer")

        if 'data' in self.queue_of_stages:
            with Logger.Paragraph('Stage [Data]' , 2):
                self.stage_data()

        if 'fit' in self.queue_of_stages:  
            with Logger.Paragraph('Stage [Fit]' , 2):
                self.stage_fit()

        if 'test' in self.queue_of_stages: 
            with Logger.Paragraph('Stage [Test]' , 2):
                self.stage_test()
        
        self.on_summarize_model()

        return self

    def go(self):
        '''alias of main_process'''
        return self.main_process()

    def stage_data(self):
        '''stage of loading model data'''
        self.on_before_data_start()
        self.on_data_start()
        self.data.load_data()
        self.on_data_end()
        self.on_after_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.config.log_operation('fit' , 'start')
        self.on_before_fit_start()
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
        self.on_after_fit_end()
        self.config.log_operation('fit' , 'end')

    def stage_test(self):
        '''stage of testing'''
        self.config.log_operation('test' , 'start')
        self.on_before_test_start()
        self.on_test_start()
        for self.status.model_date , self.status.model_num in self.iter_model_num_date():
            if self.status.model_num == 0:
                self.on_test_model_date_start()
            self.on_test_model_start()
            self.model.test()
            self.on_test_model_end()
            if self.status.model_num == self.config.model_num_list[-1]:
                self.on_test_model_date_end()
        self.on_before_test_end()
        self.on_test_end()
        self.on_after_test_end()
        self.config.log_operation('test' , 'end')

    def iter_model_num_date(self): 
        '''iter of model_date and model_num , considering is_resuming'''
       
        model_iter = list(itertools.product(self.data.model_date_list , self.config.model_num_list))
        assert self.status.stage in ['fit' , 'test'] , self.status.stage
        num_all_models = len(model_iter)
        iter_info = f'In stage [{self.status.stage}], number of all models (model_date x model_num) is {num_all_models}, '
        if self.config.is_resuming:
            if self.status.stage == 'fit':
                models_trained = np.full(len(model_iter) , True , dtype = bool)
                for i , (model_date , model_num) in enumerate(model_iter):
                    if not self.deposition.exists(model_num , model_date):
                        models_trained[max(i,0):] = False
                        break
                condition = ~models_trained
            elif self.status.stage == 'test':
                resumed_models_finished = self.record.resumed_models.groupby(['model_date' , 'model_num']).groups
                resumed = [(model_date , model_num) not in resumed_models_finished for model_date , model_num in model_iter]
                condition = np.array(resumed)
            else:
                Logger.error(f'Invalid stage for resuming iter_model_num_date: {self.status.stage}')
                condition = np.full(len(model_iter) , True , dtype = bool)
            model_iter = FilteredIterable(model_iter , condition)
            iter_info += f'resuming {num_all_models - sum(condition)} models, {sum(condition)} to go!'
        #elif self.status.stage == 'test' and self.status.fitted_model_num <= 0:
        #    model_iter = []
        else:
            iter_info += f'{num_all_models} to go!'
        Logger.note(iter_info , vb_level = 2)
        return model_iter

    def iter_model_submodels(self):
        assert self.status.stage != 'fit' , f'{self.status.stage} is not allowed to iter model submodels'
        for self.status.model_submodel in self.model_submodels: 
            self.on_test_submodel_start()
            yield self.status
            self.on_test_submodel_end()

    def iter_fit_epoches(self):
        while not self.status.fit_loop_breaker:
            self.on_fit_epoch_start()
            yield self.status
            self.on_before_fit_epoch_end()
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
        self.writer = TBSummaryWriter(self.model_tensorboad_dir)
        self.data.setup('fit' , self.model_param , self.model_date)
        self.model.new_model()
        self.metrics.new_model(self.model , self.model.complete_model_param , **self.status.status)
        self.metrics.new_attempt(**self.status.status)

    def on_fit_model_end(self): 
        if self.status.update_best_attempt(self.metrics): 
            self.model.stack_model()
        self.model.dump_model()
        self.metrics.collect_attempt()
        self.metrics.collect_model()
        
    def on_fit_epoch_start(self): ...

    def on_fit_epoch_end(self): ...

    def on_train_epoch_start(self):
        self.metrics.new_epoch(**self.status.status)

    def on_train_epoch_end(self):
        self.metrics.collect_epoch()
        
    def on_validation_epoch_start(self):
        self.metrics.new_epoch(**self.status.status)

    def on_validation_epoch_end(self):
        self.metrics.collect_epoch()
        
    def on_test_model_start(self):
        self.data.setup('test' , self.model_param , self.model_date)

    def on_test_submodel_start(self):
        self.model.load_model(submodel=self.model_submodel)
        self.metrics.new_all(self.model , self.model.complete_model_param , **self.status.status)
        
    def on_test_submodel_end(self): 
        self.metrics.collect_epoch()
        self.metrics.collect_attempt()
        self.metrics.collect_model()

    def on_test_batch_start(self):
        self.assert_date_equity()

    def assert_date_equity(self): 
        date0 = self.batch_dates[self.batch_idx] 
        date1 = self.batch_input.date0 
        if not date0 == date1:
            Logger.alert1(f'y_date: {self.data.y_date}')
            Logger.alert1(f'batch_idx: {self.batch_idx}')
            Logger.alert1(f'batch_dates: {self.batch_dates}')
            Logger.alert1(f'early_test_dates: {self.data.early_test_dates}')
            Logger.alert1(f'model_test_dates: {self.data.model_test_dates}')
            Logger.alert1(f'batch_input.i: {self.batch_input.i[0]}')
            Logger.alert1(f'batch_input.date0: {self.batch_input.date0}')
            Logger.error(f'Date equity assertion failed: {date0} != {date1}')
            raise ValueError(f'Date equity assertion failed: {date0} != {date1}')

    @property
    def penalty_kwargs(self): return {}
    
    @staticmethod
    def available_modules(module_type : Literal['nn' , 'boost' , 'all'] = 'all'):
        return AlgoModule.available_modules(module_type)
    @staticmethod
    def available_models(include_short_test : bool = False , include_factors : bool = False):
        root_paths = [PATH.model_nn , PATH.model_boost]
        if include_short_test:
            root_paths.append(PATH.model_st)
        if include_factors:
            root_paths.append(PATH.model_factor)
        bases = [f'{root.name}@{model.name}' for root in root_paths for model in root.iterdir() if model.is_dir() and not model.name.startswith('.')]
        return bases

class ModelStreamLineWithTrainer(ModelStreamLine):
    def bound_with_trainer(self , trainer): 
        self.trainer : BaseTrainer | Any = trainer
        return self

    @property
    def config(self): return self.trainer.config
    @property
    def status(self):  return self.trainer.status
    @property
    def metrics(self):  return self.trainer.metrics
    @property
    def checkpoint(self): return self.trainer.checkpoint
    @property
    def deposition(self): return self.trainer.deposition
    @property
    def device(self): return self.trainer.device
    @property
    def base_path(self): return self.trainer.base_path
    @property
    def data(self): return self.trainer.data
    @property
    def batch_input(self): return self.trainer.batch_input
    @property
    def batch_idx(self): return self.trainer.batch_idx
    @property
    def batch_output(self): return self.trainer.batch_output
    @batch_output.setter
    def batch_output(self , value : BatchOutput): 
        self.trainer.batch_output = value
    @property
    def batch_data(self): return self.trainer.batch_data
    @property
    def model_date(self): return self.trainer.model_date
    @property
    def model_num(self): return self.trainer.model_num
    @property
    def model_submodel(self): return self.trainer.model_submodel
    @property
    def model_str(self): return self.trainer.model_str