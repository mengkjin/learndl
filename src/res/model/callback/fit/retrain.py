"""
Callback to retrain if nan loss, exit too early, or get a very low RankIC
"""
from __future__ import annotations

from src.proj.bases import FittingEventType
from src.res.model.util import BaseCallBack

__all__ = ['BadAttemptRetrain']

class BadAttemptRetrain(BaseCallBack):
    """Retrain if nan loss, exit too early / moderately early, or get a very low RankIC.

    Tiered by Valid peak epoch (``effective_epoch``):
    1. ``effective_epoch <= early_exit``: always retrain.
    2. ``effective_epoch <= moderate_early_exit`` and attempt best IC ``< moderate_min_ic``: retrain.
    3. Attempt best IC and global best IC both ``< min_ic``: retrain.
    """
    CB_KEY_PARAMS = [
        'early_exit' , 'moderate_early_exit' , 'moderate_min_ic' ,
        'min_ic' , 'max_attempt' , 'max_nan_redo' ,
    ]
    def __init__(self, 
        trainer , 
        early_exit = 10 ,
        moderate_early_exit = 20 ,
        moderate_min_ic = 0.1 ,
        min_ic = 0.05 , 
        max_attempt = 4 , max_nan_redo = 4 ,
        lr_multiplier = [1 , 0.3 , 0.1 , 0.03 , 0.01] , **kwargs
    ) -> None:
        super().__init__(trainer , **kwargs)
        self.early_exit = early_exit
        self.moderate_early_exit = moderate_early_exit
        self.moderate_min_ic = moderate_min_ic
        self.min_ic = min_ic
        self.max_attempt = max_attempt
        self.max_nan_redo = max_nan_redo
        self.lr_multiplier = lr_multiplier        

    @property
    def _end_effective_epoch(self) -> int | None:
        ev = self.status.end_attempt_event
        return None if not ev else ev.effective_epoch

    @property
    def _attempt_best_ic(self) -> float | None:
        best_epoch = self.metrics.attempt_metrics.best_epoch()
        return None if not best_epoch else float(best_epoch.valid_rankic)

    @property
    def is_early_exit(self):
        ep = self._end_effective_epoch
        return ep is not None and ep <= self.early_exit

    @property
    def is_moderate_early_exit(self):
        ep = self._end_effective_epoch
        ic = self._attempt_best_ic
        if ep is None or ic is None:
            return False
        if ep <= self.early_exit:
            return False
        return ep <= self.moderate_early_exit and ic < self.moderate_min_ic

    @property
    def is_low_ic(self):
        best_epoch = self.metrics.attempt_metrics.best_epoch()
        best_attempt_ic = self.metrics.model_metrics.best_ic
        low_ic = best_epoch and best_epoch.valid_rankic < self.min_ic and best_attempt_ic < self.min_ic
        return low_ic

    @property
    def next_attempt_lr_multiplier(self):
        idx = min(self.status.attempt + 1, len(self.lr_multiplier) - 1)
        return self.lr_multiplier[idx]

    def on_fit_model_start(self):
        self.remain_nan_redo = self.max_nan_redo
    def on_train_epoch_end(self):
        self.is_nanloss = self.metrics.epoch_train_metrics.nanloss

    def on_fit_epoch_end_before(self):
        if not self.is_nanloss and (not self.status.loop_end or self.status.attempt >= self.max_attempt):
            return
        if self.is_nanloss and self.remain_nan_redo <= 0:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')
        if self.is_nanloss:
            if self.model.param_has_nan:
                self.logger.warning(f'Encounter Nan Loss, redo current attempt {self.status.attempt}! Remaining {self.remain_nan_redo} chances.')
            elif self.model.grad_has_nan:
                self.logger.warning(f'Encounter Nan Grad, redo current attempt {self.status.attempt}! Remaining {self.remain_nan_redo} chances.')
            elif self.batch_input.x_has_nan:
                self.logger.alert2('This is a bug, possible reasons include:')
                self.logger.alert2('1. Prenormer.prenorm introduce new nan after valid_position calculation' , idt = 1)
                self.logger.alert2('2. DataOperator.rolling_rotation does not match valid_position calculation (unlikely)' , idt = 1)
                self.logger.alert2('3. DataOperator.finite_position wrong calculation (almost impossible)' , idt = 1)
                raise ValueError('Encounter Nan Loss, but parameters and gradients have no nan, batch_input has nan to cause nan loss!')
            else:
                raise ValueError('Encounter Nan Loss for unknown reason!')
                    
            self.remain_nan_redo -= 1
            message = f'Get nanloss for {self.texts.attempt_key}. Redo current attempt {self.status.attempt}'
            self.trigger_retrain(FittingEventType.REDO_ATTEMPT , 'nanloss' , message)
        elif self.is_early_exit:
            message = f'Exit too early for {self.texts.attempt_key}. Start new attempt {self.status.attempt+1} with lr multiplier {self.next_attempt_lr_multiplier}'
            self.trigger_retrain(FittingEventType.NEW_ATTEMPT , 'early_exit' , message , self.next_attempt_lr_multiplier)
        elif self.is_moderate_early_exit:
            message = (
                f'Moderate early exit for {self.texts.attempt_key} '
                f'(peak_ep<={self.moderate_early_exit}, IC<{self.moderate_min_ic}). '
                f'Start new attempt {self.status.attempt+1} with lr multiplier {self.next_attempt_lr_multiplier}'
            )
            self.trigger_retrain(FittingEventType.NEW_ATTEMPT , 'moderate_early_exit' , message , self.next_attempt_lr_multiplier)
        elif self.is_low_ic:
            message = f'Get a very low RankIC for {self.texts.attempt_key}. Start new attempt {self.status.attempt+1} with lr multiplier {self.next_attempt_lr_multiplier}'
            self.trigger_retrain(FittingEventType.NEW_ATTEMPT , 'low_ic' , message , self.next_attempt_lr_multiplier)

    def trigger_retrain(self , event_type : FittingEventType , reason : str , message : str = '' , new_lr_multiplier : float = 1.):
        assert event_type in [FittingEventType.NEW_ATTEMPT , FittingEventType.REDO_ATTEMPT] , f'Invalid event type: {event_type}'
        self.model.stack_model()
        self.status.add_epoch_event(event_type , reason , message = message)
        self.trainer.new_attempt('attempt' , lr_multiplier = new_lr_multiplier)
