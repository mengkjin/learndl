from src.proj import Logger
from src.res.model.util import BaseCallBack

class BadAttemptRetrain(BaseCallBack):
    '''retrain with new lr if fitting stopped too early'''
    CB_KEY_PARAMS = ['early_exit' , 'min_ic' , 'max_attempt']
    def __init__(self, trainer , early_exit = 10 , min_ic = 0.05 , max_attempt = 4 , lr_multiplier = [1 , 0.1 , 10 , 0.01 , 100] , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.early_exit = early_exit
        self.min_ic = min_ic
        self.max_attempt = max_attempt
        self.lr_multiplier = lr_multiplier        

    @property
    def is_early_exit(self):
        return self.status.end_attempt_event.effective_epoch <= self.early_exit

    @property
    def is_low_ic(self):
        best_epoch = self.metrics.attempt_metrics.best_epoch()
        best_attempt_ic = self.metrics.model_metrics.best_ic
        return best_epoch and best_epoch.valid_rankic < self.min_ic and best_attempt_ic < self.min_ic

    def on_before_fit_epoch_end(self):
        if self.status.attempt >= self.max_attempt or not self.status.loop_end:
            return
        if self.is_early_exit or self.is_low_ic:
            self.metrics.collect_attempt()
            self.model.stack_model()
            reason = 'early_exit' if self.is_early_exit else 'low_ic'
            message = f'{self.trainer.texts.progress}, {reason}, Next attempt goes!'
            self.status.add_epoch_event('new_attempt' , reason , message = message)
            self.model.new_model(lr_multiplier = self.lr_multiplier[:self.status.attempt+1][-1])
            self.metrics.new_attempt(**self.status.status)
            
class NanLossRetrain(BaseCallBack):
    '''retrain if fitting encounters nan loss'''
    CB_KEY_PARAMS = ['max_attempt']
    def __init__(self, trainer , max_attempt = 4 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.max_attempt = max_attempt

    def on_fit_model_start(self):
        self.remain_nan_life = self.max_attempt
    def on_train_epoch_end(self):
        self.is_nanloss = self.metrics.epoch_train_metrics.nanloss
    def on_before_fit_epoch_end(self):
        if not self.is_nanloss:
            pass
        elif self.remain_nan_life > 0:
            Logger.warning(f'Initialize a new model to retrain! Lives remaining {self.remain_nan_life}')
            self.remain_nan_life -= 1
            self.metrics.collect_attempt()
            reason = 'nanloss'
            message = f'{self.trainer.texts.model} {self.trainer.texts.attempt} {self.trainer.status.epoch_key} got nanloss!'
            self.status.add_epoch_event('redo_attempt' , reason , message = message)
            self.model.new_model()
            self.metrics.new_attempt(**self.status.status)
            
        else:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')