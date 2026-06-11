from __future__ import annotations
import numpy as np
from typing import Literal

from src.proj.bases import FittingEventType
from src.res.model.util.trainer.status import EpochRecord
from src.res.model.util import BaseCallBack

def arr_plateau(arr , n : int , eps = 0.) -> bool:
    """Last n element of arr are all smaller than the previous one"""
    return arr_peaked(arr , n) or arr_converge(arr , n)

def arr_peaked(arr , n : int) -> bool:
    """Last n element of arr are all smaller than the previous one"""
    if len(arr) <= n:
        return False
    arr = arr[-(n + 1):]
    return max(arr[1:]) < arr[0]

def arr_converge(arr , n : int , tolerance = 1e-4) -> bool:
    """Last n element of arr are running within tolerance of norm"""
    hist_norm = np.sqrt(np.mean(np.square(arr)))
    if len(arr) < n:
        return False
    arr = arr[-n:]
    return max(arr) - min(arr) < hist_norm * tolerance

class EarlyStoppage(BaseCallBack):
    """Early Stoppage of Fitting, Peaked / Converged Valid Accuracy or Train Loss"""
    CB_KEY_PARAMS = ['peak_patience' , 'converge_patience' , 'converge_dataset']
    def __init__(self , trainer , peak_patience = 20 , converge_patience = 5 , converge_dataset : Literal['valid' , 'any'] = 'valid' , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.peak_patience = peak_patience
        self.converge_patience = converge_patience
        self.converge_dataset = converge_dataset

    def reset_record(self):
        self.peak_epoch = self.status.current
        self.peak_epoch_metrics = None

    def check_early_stoppage(self):
        latest_epoch_metrics = self.metrics.attempt_metrics.latest_epoch()
        if self.metrics.compare_epochs(latest_epoch_metrics, self.peak_epoch_metrics): 
            self.peak_epoch = self.status.current
            self.peak_epoch_metrics = latest_epoch_metrics

        if self.peak_epoch and (self.status.epoch - self.peak_epoch.epoch >= self.peak_patience):
            self.add_stop_event('Valid Peaked' , self.peak_epoch)
        else:
            valid_converged = arr_converge(self.metrics.attempt_metrics.total_accuracies, self.converge_patience)
            train_converged = arr_converge(self.metrics.attempt_metrics.total_losses, self.converge_patience)
            if valid_converged:
                self.add_stop_event('Valid Converged' , self.status[-self.converge_patience])
            elif self.converge_dataset == 'any' and train_converged:
                self.add_stop_event('Train Converged' , self.status[-self.converge_patience])
    def add_stop_event(self , reason : str , effect_epoch : EpochRecord):
        self.status.add_epoch_event(
            FittingEventType.END_ATTEMPT , reason.replace(' ' , '') , epoch = effect_epoch.epoch , 
            message = f'Early stoppage due to {reason} at {effect_epoch.epoch_key}, recognized at {self.status.epoch_key}'
        )

    def on_fit_model_start(self):
        self.reset_record()

    def on_validation_epoch_end(self):
        self.check_early_stoppage()