"""
Callback to control parameters of trainer
- LearnRateReset : learn rate scheduler and reset
"""
from __future__ import annotations
from copy import deepcopy

from src.proj.bases import FittingEventType
from src.res.model.util import BaseCallBack , Optimizer

__all__ = ['LearnRateReset']

class LearnRateReset(BaseCallBack):
    """Learn Rate Scheduler of Periodic Reset"""
    CB_KEY_PARAMS = ['num_reset' , 'trigger' , 'recover_level' , 'speedup2x']
    reset_speedup_param_list = ['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']
    def __init__(self, trainer, num_reset = 2 , trigger = 40 , recover_level = 1. , speedup2x = True , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.num_reset = num_reset
        self.trigger = trigger
        self.recover_level = recover_level 
        self.speedup2x = speedup2x
        self.trigger_intvl = max(trigger // 2 , 1) if speedup2x else trigger
    @property
    def optimizer(self) -> Optimizer: 
        return self.model.optimizer
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
        info = f'Reset learn rate and scheduler at {self.status.epoch_key} , effective at next epoch'
        if self.speedup2x: 
            info += ', and will speedup2x'
        self.status.add_epoch_event(FittingEventType.LOGGING , 'reset_learn_rate' , message = info)