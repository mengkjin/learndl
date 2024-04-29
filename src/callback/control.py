import numpy as np
import pandas as pd

import time
import torch
from .base import BasicCallBack , WithCallBack
from .. import util as U
from ..func import list_converge
class EarlyStoppage(BasicCallBack):
    def __init__(self , model_module , patience = 20) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
    def on_fit_model_start(self):
        self._epoch_best = -1
        self._score_best = -10000.
    def on_validation_epoch_end(self):
        if self.metrics.valid_scores[-1] > self._score_best: 
            self._epoch_best = self.status.epoch 
            self._score_best = self.metrics.valid_scores[-1]
        if self.status.epoch - self._epoch_best >= self._patience:
            self.status.end_of_loop.add_status('EarlyStop' , self._epoch_best)

class ValidationConverge(BasicCallBack):
    def __init__(self , model_module , patience = 5 , eps = 1.0e-5) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
        self._eps      = eps
    def on_validation_epoch_end(self):
        if list_converge(self.metrics.metric_epochs['valid.score'], self._patience , self._eps):
            self.status.end_of_loop.add_status('Valid Cvg' , self.status.epoch - self._patience + 1)

class TrainConverge(BasicCallBack):
    def __init__(self , model_module , patience = 5 , eps = 1.0e-5) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
        self._eps      = eps
    def on_validation_epoch_end(self):
        if list_converge(self.metrics.metric_epochs['train.loss'], self._patience , self._eps):
            self.status.end_of_loop.add_status('Train Cvg' , self.status.epoch - self._patience + 1)

class FitConverge(BasicCallBack):
    def __init__(self , model_module , patience = 5 , eps = 1.0e-5) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
        self._eps      = eps
    def on_fit_epoch_end(self):
        if (list_converge(self.metrics.metric_epochs['train.loss'], self._patience , self._eps) and 
            list_converge(self.metrics.metric_epochs['valid.score'], self._patience , self._eps)):
            self.status.end_of_loop.add_status('T & V Cvg' , self.status.epoch - self._patience + 1)

class EarlyExitRetrain(BasicCallBack):
    def __init__(self, model_module , earliest = 20 , max_attempt = 4) -> None:
        super().__init__(model_module)
        self._earliest = earliest
        self._max_attempt = max_attempt
        self._print_info()
    def on_fit_model_start(self):
        self.status.attempt = 0
    def on_validation_epoch_end(self):
        if (self.status.end_of_loop and 
            self.status.end_of_loop.trigger_ep <= self._earliest 
            and self.status.attempt < self._max_attempt):
            self.metrics.new_attempt()
            self.status.new_attempt()
            self.module.checkpoint.new_model(self.module.model_param , self.status.model_date)
            self.module.load_model(True)
            self.status.attempt += 1
            self.status.add_event('new_attempt')

class NanLossRetrain(BasicCallBack):
    '''Deal with nanloss, life -1 and change nanloss condition to True'''
    def __init__(self, model_module , max_attempt = 4) -> None:
        super().__init__(model_module)
        self._print_info()
        self._max_attempt = max_attempt
    def on_fit_model_start(self):
        self._nanlife = self._max_attempt
    def on_train_epoch_end(self):
        self._nanloss = self.metrics.metric_batchs.nanloss
    def on_fit_epoch_end(self):
        if not self._nanloss:
            pass
        elif self._nanlife > 0:
            self.logger.error('Model {model_date}.{model_num} Attempt{attempt}, epoch{epoch} got nanloss!'.format(**self.status.__dict__))
            self.logger.error(f'Initialize a new model to retrain! Lives remaining {self._nanlife}')
            self.metrics.new_attempt()
            self.status.new_attempt()
            self.module.checkpoint.new_model(self.module.model_param , self.status.model_date)
            self.module.load_model(True)
            self._nanlife -= 1
        else:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')

class CudaEmptyCache(BasicCallBack):
    def __init__(self , model_module , batch_interval = 20) -> None:
        super().__init__(model_module)
        self._print_info('pretty slow')
        self._interval = batch_interval
        # 2.5s for 86 epochs
        
    def _empty_cache(self):
        if (self.module.batch_idx + 1) % self._interval == 0 : torch.cuda.empty_cache()
    def on_train_batch_end(self):        self._empty_cache()
    def on_validation_batch_end(self):   self._empty_cache()
    def on_test_batch_end(self):         self._empty_cache()

class ProcessTimer(WithCallBack):
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        self._print_info()
        self._pt = {}
    def __enter__(self):
        super().__enter__()
        self.start_time = time.time()
    def __exit__(self): 
        if self.hook_name not in self._pt.keys(): self._pt[self.hook_name] = []
        self._pt[self.hook_name].append(time.time() - self.start_time)
    def on_summarize_model(self):
        tb = pd.DataFrame([[k , len(v) , np.sum(v) , np.mean(v)] for k,v in self._pt.items()] ,
                          columns = ['keys' , 'num_calls', 'total_time' , 'avg_time'])
        print(tb.sort_values(by=['total_time'],ascending=False))