from __future__ import annotations


import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime
from typing import Any , Literal

from src.proj import Logger
from .metrics import Metrics
from .streamline import ModelStreamLine

@dataclass
class _EndEpochStamp:
    """End epoch stamp class, used to store the end epoch of the model stream line"""
    name  : str
    epoch : int # epoch of trigger

class _FitLoopBreaker:
    """Fit loop breaker class, used to break the fit loop when the epoch is too large or meet some conditions"""
    def __init__(self , max_epoch : int = 200):
        self.max_epoch = max_epoch
        self.status : list[_EndEpochStamp] = []
    def __bool__(self): 
        return len(self.status) > 0
    def __repr__(self): 
        return f'{self.__class__.__name__}(max_epoch={self.max_epoch},status={self.status})'  
    def new_loop(self): 
        self.status = []
    def loop_end(self , epoch):
        if epoch >= self.max_epoch - 1: 
            self.add_status('Max Epoch' , epoch)
    def add_status(self , status : str , epoch : int): 
        self.status.append(_EndEpochStamp(status , epoch))
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

class TrainerStatus(ModelStreamLine):
    """Trainer status class, used to store the status of the trainer"""
    def __init__(self , max_epoch : int = 200):
        self.max_epoch : int = max_epoch
        self.stage   : Literal['data' , 'fit' , 'test'] = 'data'
        self.dataset : Literal['train' , 'valid' , 'test' , 'predict'] = 'train'
        self.epoch   : int = -1
        self.attempt : int = 0
        self.round   : int = 0

        self.epoch_model : int = -1
        
        self.model_num  : int = -1
        self.model_date : int = -1
        self.model_submodel : str = 'best'
        self.epoch_event : list[str] = []

        self.best_attempt : tuple[int , Any] = (-1 , None)

        self.fitted_model_num : int = 0

        self.fit_loop_breaker = _FitLoopBreaker(self.max_epoch)
        self.fit_iter_num : int = 0

        self.start_times : dict[str,datetime] = {}
        self.end_times : dict[str,datetime] = {}
        self.test_summary : pd.DataFrame = pd.DataFrame()

    def as_dict(self):
        d = {k:getattr(self,k) for k in 
             ['max_epoch' , 'stage' , 'dataset' , 'epoch' , 'attempt' , 
              'round' , 'model_num' , 'model_date' , 'model_submodel' , 
              'epoch_event' , 'best_attempt' , 'fitted_model_num']}
        return d

    def __repr__(self):
        return f'TrainerStatus({", ".join([f"{k}={v}" for k,v in self.status.items()])})'

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
    def update_best_attempt(self , metrics : Metrics):
        if metrics.better_attempt(self.best_attempt[1]):
            self.best_attempt = (self.attempt , metrics.best_epoch_metric)
            return True
        return False
    def stage_data(self): self.stage = 'data'
    def stage_fit(self):  self.stage = 'fit'
    def stage_test(self): self.stage = 'test'
    def on_before_data_start(self):    self.start_times['data'] = datetime.now()
    def on_after_data_end(self):       self.end_times['data'] = datetime.now()
    def on_before_fit_start(self):     self.start_times['fit'] = datetime.now()
    def on_after_fit_end(self):        self.end_times['fit'] = datetime.now()
    def on_before_test_start(self):    self.start_times['test'] = datetime.now()
    def on_after_test_end(self):       self.end_times['test'] = datetime.now()
    def on_train_epoch_start(self): self.dataset = 'train'
    def on_validation_epoch_start(self): self.dataset = 'valid'
    def on_test_model_start(self): self.dataset = 'test'
    def on_fit_model_start(self):
        if self.fit_iter_num == 0:
            Logger.note(f'In Stage [{self.stage}], First Iterance: ({self.model_date} , {self.model_num})')
        self.fit_iter_num += 1
        self.attempt = -1
        self.best_attempt = (-1 , None)
        self.epoch = -1
        self.epoch_model = -1
        self.new_attempt()
    def on_fit_model_end(self):
        self.fitted_model_num += 1
    def on_fit_epoch_start(self):
        self.epoch += 1
        self.epoch_model += 1
        self.epoch_event = []
    def on_fit_epoch_end(self):
        self.fit_loop_breaker.loop_end(self.epoch)
    def new_attempt(self , event : Literal['new_attempt' , 'nanloss'] = 'new_attempt'):
        self.epoch = -1
        self.round = 0
        self.epoch_event = []
        self.fit_loop_breaker.new_loop()
        self.add_event(event)
        if event == 'new_attempt': 
            self.attempt += 1        

    def add_event(self , event : str | None):
        if event: 
            self.epoch_event.append(event)
