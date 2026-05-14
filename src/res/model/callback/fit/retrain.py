from typing import Literal

from src.proj import Logger
from src.res.model.util import BaseCallBack

class BadAttemptRetrain(BaseCallBack):
    '''Retrain if nan loss, exit too early, or get a very low RankIC'''
    CB_KEY_PARAMS = ['early_exit' , 'min_ic' , 'max_attempt' , 'max_nan_redo']
    def __init__(self, 
        trainer , 
        early_exit = 10 , min_ic = 0.05 , 
        max_attempt = 4 , max_nan_redo = 4 ,
        lr_multiplier = [1 , 0.1 , 10 , 0.01 , 100] , **kwargs
    ) -> None:
        super().__init__(trainer , **kwargs)
        self.early_exit = early_exit
        self.min_ic = min_ic
        self.max_attempt = max_attempt
        self.max_nan_redo = max_nan_redo
        self.lr_multiplier = lr_multiplier        

    @property
    def is_early_exit(self):
        return bool(self.status.end_attempt_event) and self.status.end_attempt_event.effective_epoch <= self.early_exit

    @property
    def is_low_ic(self):
        best_epoch = self.metrics.attempt_metrics.best_epoch()
        best_attempt_ic = self.metrics.model_metrics.best_ic
        low_ic = best_epoch and best_epoch.valid_rankic < self.min_ic and best_attempt_ic < self.min_ic
        return low_ic

    @property
    def next_attempt_lr_multiplier(self):
        return self.lr_multiplier[:self.status.attempt+1][-1]

    def on_fit_model_start(self):
        self.remain_nan_redo = self.max_nan_redo
    def on_train_epoch_end(self):
        self.is_nanloss = self.metrics.epoch_train_metrics.nanloss

    def on_before_fit_epoch_end(self):
        if not self.is_nanloss and (not self.status.loop_end or self.status.attempt >= self.max_attempt):
            return
        if self.is_nanloss and self.remain_nan_life <= 0:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')
        if self.is_nanloss:
            Logger.warning(f'Encounter Nan Loss, redo current attempt {self.status.attempt}! Remaining {self.remain_nan_life} chances.')
            self.remain_nan_life -= 1
            message = f'{self.trainer.texts.model} {self.trainer.texts.attempt} {self.trainer.status.epoch_key} got nanloss! Redo current attempt {self.status.attempt}!'
            self.trigger_retrain('redo_attempt' , 'nanloss' , message)
        elif self.is_early_exit:
            message = f'{self.trainer.texts.progress}, exit too early. Start new attempt {self.status.attempt+1}!'
            self.trigger_retrain('new_attempt' , 'early_exit' , message , self.next_attempt_lr_multiplier)
        elif self.is_low_ic:
            message = f'{self.trainer.texts.progress}, get a very low RankIC, Start new attempt {self.status.attempt+1}!'
            self.trigger_retrain('new_attempt' , 'low_ic' , message , self.next_attempt_lr_multiplier)

    def trigger_retrain(self , event_type : Literal['new_attempt' , 'redo_attempt'] , reason : str , message : str = '' , new_lr_multiplier : float = 1.):
        self.model.stack_model()
        self.status.add_epoch_event(event_type , reason , message = message)
        self.model.new_attempt(lr_multiplier = new_lr_multiplier)