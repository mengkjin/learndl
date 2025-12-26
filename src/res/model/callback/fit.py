import torch
from copy import deepcopy

from src import math as FUNC
from src.proj import Logger
from src.res.model.util import BaseCallBack , Optimizer

class EarlyStoppage(BaseCallBack):
    '''stop fitting when validation score cease to improve'''
    CB_KEY_PARAMS = ['patience']
    def __init__(self , trainer , patience = 20 , **kwargs) -> None:
        self.patience = patience
        super().__init__(trainer , **kwargs)

    def on_fit_model_start(self):
        self.metric_best_epoch  = -1
        self.metric_best_level = None

    def on_validation_epoch_end(self):
        if self.metrics.better_epoch(self.metric_best_level): 
            self.metric_best_epoch  = self.status.epoch 
            self.metric_best_level = self.metrics.last_metric
        if self.status.epoch - self.metric_best_epoch >= self.patience:
            self.status.fit_loop_breaker.add_status('EarlyStop' , self.metric_best_epoch)

class ValidationConverge(BaseCallBack):
    '''stop fitting when valid_score converge'''
    CB_KEY_PARAMS = ['patience' , 'eps']
    def __init__(self , trainer , patience = 5 , eps = 1.0e-5 , **kwargs) -> None:
        self.patience = patience
        self.eps = eps
        super().__init__(trainer , **kwargs)

    def on_validation_epoch_end(self):
        if FUNC.list_converge(self.metrics.metric_epochs['valid.score'], self.patience , self.eps):
            self.status.fit_loop_breaker.add_status('Valid Cvg' , self.status.epoch - self.patience + 1)

class TrainConverge(BaseCallBack):
    '''stop fitting when train_loss converge'''
    CB_KEY_PARAMS = ['patience' , 'eps']
    def __init__(self , trainer , patience = 5 , eps = 1.0e-5 , **kwargs) -> None:
        self.patience = patience
        self.eps = eps
        super().__init__(trainer , **kwargs)

    def on_validation_epoch_end(self):
        if FUNC.list_converge(self.metrics.metric_epochs['train.loss'], self.patience , self.eps):
            self.status.fit_loop_breaker.add_status('Train Cvg' , self.status.epoch - self.patience + 1)

class FitConverge(BaseCallBack):
    '''stop fitting when train_loss and valid_score converge'''
    CB_KEY_PARAMS = ['patience' , 'eps']
    def __init__(self , trainer , patience = 5 , eps = 1.0e-5 , **kwargs) -> None:
        self.patience = patience
        self.eps = eps
        super().__init__(trainer , **kwargs)

    def on_validation_epoch_end(self):
        if (FUNC.list_converge(self.metrics.metric_epochs['train.loss'], self.patience , self.eps) and 
            FUNC.list_converge(self.metrics.metric_epochs['valid.score'], self.patience , self.eps)):
            self.status.fit_loop_breaker.add_status('T & V Cvg' , self.status.epoch - self.patience + 1)

class EarlyExitRetrain(BaseCallBack):
    '''retrain with new lr if fitting stopped too early'''
    CB_KEY_PARAMS = ['earliest' , 'max_attempt']
    def __init__(self, trainer , earliest = 5 , max_attempt = 4 , lr_multiplier = [1 , 0.1 , 10 , 0.01 , 100] , **kwargs) -> None:
        self.earliest = earliest
        self.max_attempt = max_attempt
        self.lr_multiplier = lr_multiplier
        super().__init__(trainer , **kwargs)

    def on_fit_model_start(self):
        self.status.attempt = 0
    def on_before_fit_epoch_end(self):
        if (self.status.fit_loop_breaker and 
            self.status.fit_loop_breaker.trigger_ep <= self.earliest 
            and self.status.attempt < self.max_attempt):
            if self.metrics.better_attempt(self.status.best_attempt_metric):
                self.status.best_attempt_metric = self.metrics.best_metric
                self.trainer.stack_model()
            self.metrics.new_attempt()
            self.status.new_attempt()
            self.model.new_model(lr_multiplier = self.lr_multiplier[:self.status.attempt+1][-1])

class NanLossRetrain(BaseCallBack):
    '''retrain if fitting encounters nan loss'''
    CB_KEY_PARAMS = ['max_attempt']
    def __init__(self, trainer , max_attempt = 4 , **kwargs) -> None:
        self.max_attempt = max_attempt
        super().__init__(trainer , **kwargs)

    def on_fit_model_start(self):
        self.remain_nan_life = self.max_attempt
    def on_train_epoch_end(self):
        self.is_nanloss = self.metrics.metric_batchs.nanloss
    def on_before_fit_epoch_end(self):
        if not self.is_nanloss:
            pass
        elif self.remain_nan_life > 0:
            Logger.warning(f'Initialize a new model to retrain! Lives remaining {self.remain_nan_life}')
            self.remain_nan_life -= 1

            self.metrics.new_attempt()
            self.status.new_attempt('nanloss')
            self.model.new_model()
        else:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')

class CudaEmptyCache(BaseCallBack):
    '''CudaEmptyCache every few batch (pretty slow)'''
    CB_KEY_PARAMS = ['batch_interval']
    def __init__(self , trainer , batch_interval = 20 , **kwargs) -> None:
        self.batch_interval = batch_interval
        super().__init__(trainer , **kwargs)
        # 2.5s for 86 epochs
    def empty_cache(self):
        if (self.trainer.batch_idx + 1) % self.batch_interval == 0 : 
            torch.cuda.empty_cache()
    def on_train_batch_end(self):        
        self.empty_cache()
    def on_validation_batch_end(self):   
        self.empty_cache()
    def on_test_batch_end(self):         
        self.empty_cache()

class ResetOptimizer(BaseCallBack):
    '''reset optimizer on some epoch (can speedup scheduler)'''
    CB_KEY_PARAMS = ['num_reset' , 'trigger' , 'recover_level' , 'speedup2x']
    reset_speedup_param_list = ['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']
    def __init__(self, trainer, num_reset = 2 , trigger = 40 , recover_level = 1. , speedup2x = True , **kwargs) -> None:
        self.num_reset = num_reset
        self.trigger = trigger
        self.recover_level = recover_level 
        self.speedup2x = speedup2x
        self.trigger_intvl = max(trigger // 2 , 1) if speedup2x else trigger
        super().__init__(trainer , **kwargs)
    @property
    def optimizer(self) -> Optimizer: 
        return self.trainer.model.optimizer
    @property
    def reset_epoch(self) -> bool:
        i = self.status.epoch + 1 - self.trigger
        return (0 <= i < self.trigger_intvl * self.num_reset) and (i % self.trigger_intvl == 0)
    def halved_param(self , param : dict):
        return {k:((v // 2) if k in self.reset_speedup_param_list else v) for k,v in param.items()}
    def on_train_epoch_end(self):
        if not self.reset_epoch: 
            return

        # confirm reset : change back optimizor learn rate
        for param_group in self.optimizer.optimizer.param_groups:
            param_group['lr'] = param_group['lr_param']  * self.recover_level
        
        # confirm reset : reassign scheduler
        shd_param = deepcopy(self.optimizer.shd_param)
        if self.speedup2x: 
            shd_param = self.halved_param(shd_param)

        self.optimizer.scheduler = self.optimizer.load_scheduler(self.optimizer.optimizer , shd_param)
        self.status.add_event('reset_learn_rate')