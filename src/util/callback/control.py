import torch

from copy import deepcopy

from .base import BasicCallBack
from ..optim import Optimizer
from ...func import list_converge

class EarlyStoppage(BasicCallBack):
    '''stop fitting when validation score cease to improve'''
    def __init__(self , model_module , patience = 20) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
    def on_fit_model_start(self):
        self._epoch_best  = -1
        self._metric_best = None
    def on_validation_epoch_end(self):
        if self.metrics.better_epoch(self._metric_best): 
            self._epoch_best  = self.status.epoch 
            self._metric_best = self.metrics.last_metric
        if self.status.epoch - self._epoch_best >= self._patience:
            self.status.end_of_loop.add_status('EarlyStop' , self._epoch_best)

class ValidationConverge(BasicCallBack):
    '''stop fitting when valid_score converge'''
    def __init__(self , model_module , patience = 5 , eps = 1.0e-5) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
        self._eps      = eps
    def on_validation_epoch_end(self):
        if list_converge(self.metrics.metric_epochs['valid.score'], self._patience , self._eps):
            self.status.end_of_loop.add_status('Valid Cvg' , self.status.epoch - self._patience + 1)

class TrainConverge(BasicCallBack):
    '''stop fitting when train_loss converge'''
    def __init__(self , model_module , patience = 5 , eps = 1.0e-5) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
        self._eps      = eps
    def on_validation_epoch_end(self):
        if list_converge(self.metrics.metric_epochs['train.loss'], self._patience , self._eps):
            self.status.end_of_loop.add_status('Train Cvg' , self.status.epoch - self._patience + 1)

class FitConverge(BasicCallBack):
    '''stop fitting when train_loss and valid_score converge'''
    def __init__(self , model_module , patience = 5 , eps = 1.0e-5) -> None:
        super().__init__(model_module)
        self._print_info()
        self._patience = patience
        self._eps      = eps
    def on_validation_epoch_end(self):
        if (list_converge(self.metrics.metric_epochs['train.loss'], self._patience , self._eps) and 
            list_converge(self.metrics.metric_epochs['valid.score'], self._patience , self._eps)):
            self.status.end_of_loop.add_status('T & V Cvg' , self.status.epoch - self._patience + 1)

class EarlyExitRetrain(BasicCallBack):
    '''retrain with new lr if fitting stopped too early'''
    def __init__(self, model_module , earliest = 5 , max_attempt = 4 , lr_multiplier = [1 , 0.1 , 10 , 0.01 , 100]) -> None:
        super().__init__(model_module)
        self._print_info()
        self._earliest = earliest
        self._max_attempt = max_attempt
        self._lr_multiplier = lr_multiplier
    def on_fit_model_start(self):
        self.status.attempt = 0
    def on_before_fit_epoch_end(self):
        if (self.status.end_of_loop and 
            self.status.end_of_loop.trigger_ep <= self._earliest 
            and self.status.attempt < self._max_attempt):
            self.module.stack_model()
            self.metrics.new_attempt()
            self.status.new_attempt()
            self.module.load_model(True , lr_multiplier = self._lr_multiplier[:self.status.attempt+1][-1])

class NanLossRetrain(BasicCallBack):
    '''retrain if fitting encounters nan loss'''
    def __init__(self, model_module , max_attempt = 4) -> None:
        super().__init__(model_module)
        self._print_info()
        self._max_attempt = max_attempt
    def on_fit_model_start(self):
        self._nanlife = self._max_attempt
    def on_train_epoch_end(self):
        self._nanloss = self.metrics.metric_batchs.nanloss
    def on_before_fit_epoch_end(self):
        if not self._nanloss:
            pass
        elif self._nanlife > 0:
            self.logger.error(f'Initialize a new model to retrain! Lives remaining {self._nanlife}')
            self._nanlife -= 1

            self.metrics.new_attempt()
            self.status.new_attempt('nanloss')
            self.module.load_model(True)
        else:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')

class CudaEmptyCache(BasicCallBack):
    '''CudaEmptyCache every few batch (pretty slow)'''
    def __init__(self , model_module , batch_interval = 20) -> None:
        super().__init__(model_module)
        self._print_info()
        self._interval = batch_interval
        # 2.5s for 86 epochs
    def _empty_cache(self):
        if (self.module.batch_idx + 1) % self._interval == 0 : torch.cuda.empty_cache()
    def on_train_batch_end(self):        self._empty_cache()
    def on_validation_batch_end(self):   self._empty_cache()
    def on_test_batch_end(self):         self._empty_cache()

class ResetOptimizer(BasicCallBack):
    '''reset optimizer on some epoch (can speedup scheduler)'''
    reset_speedup_param_list = ['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']
    def __init__(self, model_module, num_reset = 2 , trigger = 40 , recover_level = 1. , speedup2x = True) -> None:
        super().__init__(model_module)
        self._print_info()
        self._num_reset = num_reset
        self._trigger = trigger
        self._recover_level = recover_level 
        self._speedup2x = speedup2x
        self._trigger_intvl = max(self._trigger // 2 , 1) if self._speedup2x else self._trigger
    @property
    def optim(self) -> Optimizer: return self.module.optimizer
    @property
    def _reset_epoch(self) -> bool:
        i = self.status.epoch + 1 - self._trigger
        return (0 <= i < self._trigger_intvl * self._num_reset) and (i % self._trigger_intvl == 0)
    def _halved_param(self , param : dict):
        return {k:((v // 2) if k in self.reset_speedup_param_list else v) for k,v in param.items()}
    def on_train_epoch_end(self):
        if not self._reset_epoch: return

        # confirm reset : change back optimizor learn rate
        for param_group in self.optim.optimizer.param_groups:
            param_group['lr'] = param_group['lr_param']  * self._recover_level
        
        # confirm reset : reassign scheduler
        shd_param = deepcopy(self.optim.shd_param)
        if self._speedup2x: shd_param = self._halved_param(shd_param)

        self.optim.scheduler = self.optim.load_scheduler(self.optim.optimizer , shd_param)
        self.status.add_event('reset_learn_rate')

class DynamicDataLink(BasicCallBack):
    '''assign and unlink dynamic data in tra networks'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        self._print_info()
    def _net_method(self , key , *args , **kwargs): 
        if (method := getattr(self.module.net,key,None)): method(*args , **kwargs)
    def on_train_epoch_start(self):      self._net_method('dynamic_data_assign' , self.module)
    def on_validation_epoch_start(self): self._net_method('dynamic_data_assign' , self.module)
    def on_test_model_type_start(self):  self._net_method('dynamic_data_assign' , self.module)
    def on_before_save_model(self):      self._net_method('dynamic_data_unlink')