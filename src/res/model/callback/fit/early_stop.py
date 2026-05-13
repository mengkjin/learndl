from __future__ import annotations
import numpy as np
from src.res.model.util import BaseCallBack

def arr_plateau(arr , n : int , eps = 0.) -> bool:
    '''Last n element of arr are all smaller than the previous one'''
    return arr_peaked(arr , n) or arr_converge(arr , n)

def arr_peaked(arr , n : int) -> bool:
    '''Last n element of arr are all smaller than the previous one'''
    if len(arr) <= n:
        return False
    arr = arr[-(n + 1):]
    return max(arr[1:]) < arr[0]

def arr_converge(arr , n : int , tolerance = 1e-4) -> bool:
    '''Last n element of arr are running within tolerance of norm'''
    hist_norm = np.sqrt(np.mean(np.square(arr)))
    if len(arr) < n:
        return False
    arr = arr[-n:]
    return max(arr) - min(arr) < hist_norm * tolerance

class EarlyStoppage(BaseCallBack):
    '''stop fitting when validation accuracy cease to improve'''
    CB_KEY_PARAMS = ['patience']
    def __init__(self , trainer , patience = 20 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.patience = patience

    def reset_record(self):
        self.best_epoch = None

    def check_early_stoppage(self):
        latest_epoch = self.metrics.attempt_metrics.latest_epoch()
        if self.metrics.compare_epochs(latest_epoch, self.best_epoch): 
            self.best_epoch = latest_epoch
        if not self.best_epoch:
            return
        if self.status.epoch - self.best_epoch.epoch >= self.patience:
            self.status.add_epoch_event(
                'end_attempt' , 'EarlyStop' , epoch = self.best_epoch.epoch , 
                message = f'Early stoppage at epoch {self.best_epoch.epoch}, force end attempt at epoch {self.status.epoch}'
            )

    def on_fit_model_start(self):
        self.reset_record()

    def on_validation_epoch_end(self):
        self.check_early_stoppage()

class ValidationConverge(BaseCallBack):
    '''stop fitting when valid_accuracy converge'''
    CB_KEY_PARAMS = ['patience']
    def __init__(self , trainer , patience = 5 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.patience = patience

    def on_validation_epoch_end(self):
        if arr_converge(self.metrics.attempt_metrics.total_accuracies, self.patience):
            self.status.add_epoch_event(
                'end_attempt' , 'Valid Cvg' , epoch = self.status.epoch - self.patience + 1 , 
                message = f'Valid accuracy converged at epoch {self.status.epoch - self.patience + 1}, force end attempt at epoch {self.status.epoch}'
            )

class TrainConverge(BaseCallBack):
    '''stop fitting when train_loss converge'''
    CB_KEY_PARAMS = ['patience']
    def __init__(self , trainer , patience = 5 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.patience = patience

    def on_validation_epoch_end(self):
        if arr_converge(self.metrics.attempt_metrics.total_losses, self.patience):
            self.status.add_epoch_event(
                'end_attempt' , 'Train Cvg' , epoch = self.status.epoch - self.patience + 1 , 
                message = f'Train loss converged at epoch {self.status.epoch - self.patience + 1}, force end attempt at epoch {self.status.epoch}'
            )

class FitConverge(BaseCallBack):
    '''stop fitting when train_loss and valid_accuracy converge'''
    CB_KEY_PARAMS = ['patience']
    def __init__(self , trainer , patience = 5 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.patience = patience

    def on_validation_epoch_end(self):
        if (arr_converge(self.metrics.attempt_metrics.total_losses, self.patience) and 
            arr_converge(self.metrics.attempt_metrics.total_accuracies, self.patience)):
            self.status.add_epoch_event(
                'end_attempt' , 'T & V Cvg' , epoch = self.status.epoch - self.patience + 1 , 
                message = f'Train loss and valid accuracy converged at epoch {self.status.epoch - self.patience + 1}, force end attempt at epoch {self.status.epoch}'
            )