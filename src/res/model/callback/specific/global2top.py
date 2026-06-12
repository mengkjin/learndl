"""
Callback specifically for Global2Top module
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from typing import Literal , Callable
from src.proj.bases import FittingEventType
from src.res.model.util import BaseCallBack
from src.res.model.util.core import epoch_key

__all__ = ['SpecificCB_Global2Top']

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

def sigmoid_curve(i, min_value = 0., max_value = 1., steepness=3., midpoint=10.0):
    """
    Returns a sigmoid-shaped value starting near 0 and asymptoting to 0.5.
    
    Args:
        i: The index or input value.
        steepness: How fast the curve rises.
        midpoint: The index where the value is exactly max_value / 2.
    """
    return (max_value - min_value) / (1 + np.exp(-steepness * (i / midpoint - 1))) + min_value

def split_accuracies(accuracies : dict[str,float]) -> tuple[float,float]:
    glb = [accu for name , accu in accuracies.items() if name.startswith('global.')]
    top = [accu for name , accu in accuracies.items() if name.startswith('top.')]
    return sum(glb), sum(top)

def synthetic_accuracy(glb , top , * , glb_climax : float | None = None , glb_multiplier : float = 5.):
    if glb_climax is None:
        return top + glb * glb_multiplier
    return top - np.maximum(0 , glb_climax - glb) * glb_multiplier

def aggregator_factory(phase : Literal['global' , 'top'] , glb_climax : float | None = None , glb_multiplier : float = 5.) -> Callable[[dict[str,float]],float]:
    if phase == 'global':
        def _aggregator(accuracies : dict[str,float]) -> float:
            glb , _ = split_accuracies(accuracies)
            return glb
    elif phase == 'top':
        def _aggregator(accuracies : dict[str,float]) -> float:
            glb , top = split_accuracies(accuracies)
            return synthetic_accuracy(glb , top , glb_climax = glb_climax , glb_multiplier = glb_multiplier)
    else:
        raise ValueError(f'Invalid phase: {phase}')
    return _aggregator

class SpecificCB_Global2Top(BaseCallBack):
    """Fitting Control of Global2Top"""
    CB_KEY_PARAMS = ['protect' , 'plateau_patience' , 'converge_patience' , 'top_converge_alpha']
    OverrideCallbacks = ['EarlyStoppage']
    ConflictModuleTypes = ['factor']
    def __init__(self , trainer , 
        protect : int = 10 , 
        plateau_patience : int = 20 , 
        converge_patience : int = 10 ,
        top_converge_alpha : float = 0.6 ,**kwargs) -> None:
        super().__init__(trainer , **kwargs)
        assert 0 <= top_converge_alpha <= 1 , f'top_converge_alpha must be between 0 and 1 and is {top_converge_alpha}'
        self.protect = protect
        self.plateau_patience = plateau_patience
        self.converge_patience = converge_patience
        self.top_converge_alpha = top_converge_alpha

    def override_configuration(self):
        if (max_epoch := self.config.max_epoch) <= self.protect:
            self.config.model_config['train.max_epoch'] = max_epoch + self.protect
            self.logger.alert1(f'overrides [max_epoch] {max_epoch} -> {max_epoch + self.protect}' , idt = 1 , vb = 1)
            
        loss_criteria = self.config.criterion_loss
        if tuple(loss_criteria.keys()) != ('global2top',):
            new_loss_criteria = {k:v for k,v in self.config.criterion_loss.items() if k == 'global2top'}
            self.config.model_config['train.criterion.loss'] = new_loss_criteria or {'global2top':{}}
            self.logger.alert1(f'overrides [loss criteria] {loss_criteria} -> {self.config.criterion_loss}' , idt = 1 , vb = 1)

        accuracy_criteria = self.config.criterion_accuracy
        if tuple(accuracy_criteria.keys()) != ('global2top',):
            new_accuracy_criteria = {k:v for k,v in self.config.criterion_accuracy.items() if k == 'global2top'}
            self.config.model_config['train.criterion.accuracy'] = new_accuracy_criteria or {'global2top':{}}
            self.logger.alert1(f'overrides [accuracy criteria] {accuracy_criteria} -> {self.config.criterion_accuracy}' , idt = 1 , vb = 1)

    def reset_record(self):
        self.valid_accuracies = pd.DataFrame()
        self.global_plateaued = False
        self.global_climax_epoch = -1
        self.global_climax_level = -1
        self.global_support_level = -1
        self.loss_weights_set = False
        self.accuracy_computer_set = False
        self.fitting_phase : Literal['global' , 'top'] = 'global'
        self.set_accuracy_verdict()

    def collect_accuracies(self):
        accuracies = pd.concat(self.metrics.attempt_metrics.tables['valid_epoch_accuracies'])
        phases = accuracies.index.get_level_values('phase').unique()
        if len(phases) > 1:
            phase_accu_list : list[pd.DataFrame] = []
            max_epoch = accuracies.index.get_level_values('epoch').max()
            for phase in phases[::-1]:
                phase_accu = accuracies.query(f'phase == {phase} & epoch <= {max_epoch}')
                if not phase_accu.empty:
                    max_epoch = phase_accu.index.get_level_values('epoch').min() - 1
                    phase_accu_list.append(phase_accu)
            self.valid_accuracies = pd.concat(phase_accu_list).sort_index()
        else:
            self.valid_accuracies = accuracies
        self.glb_accus = [col for col in self.valid_accuracies.columns if col.startswith('global.')]
        self.top_accus = [col for col in self.valid_accuracies.columns if col.startswith('top.')]
        
    @property
    def global_accuracies(self) -> np.ndarray:
        return self.valid_accuracies.loc[:,self.glb_accus].sum(axis=1).to_numpy()
    @property
    def top_accuracies(self) -> np.ndarray:
        return self.valid_accuracies.loc[:,self.top_accus].sum(axis=1).to_numpy()

    def check_global_plateau(self):
        """check if the global accuracy is plateauing"""
        if self.status.epoch < self.protect or self.global_plateaued or len(self.valid_accuracies) <= self.plateau_patience:
            return
        if arr_plateau(self.global_accuracies , n = self.plateau_patience):
            self.global_plateaued = True
            best_epoch = self.metrics.attempt_metrics.best_epoch()
            assert best_epoch , f'best epoch is not found for {self.metrics.attempt_metrics}'
            self.global_climax_epoch = best_epoch.epoch
            self.global_climax_level = sum([accu for name , accu in best_epoch.valid_accuracies.items() if name.startswith('global.')])
            self.global_support_level = max(0 , min(self.global_climax_level * 0.9 , self.global_climax_level - 0.01))

    def check_top_converge(self):
        if self.fitting_phase == 'global' or len(self.valid_accuracies) <= self.converge_patience + self.global_climax_epoch:
            return
        overall_accuracies = synthetic_accuracy(self.global_accuracies , self.top_accuracies , glb_climax = self.global_climax_level or 0.)
        if arr_plateau(overall_accuracies , n = self.converge_patience) or all(self.global_accuracies[-self.converge_patience:] < self.global_support_level):
            start_len = len(overall_accuracies) - self.converge_patience - 1
            self.metric_best_epoch = overall_accuracies[start_len:].argmax().item() + start_len
            self.metric_best_level = self.valid_accuracies.iloc[self.metric_best_epoch].to_dict()
            self.status.add_epoch_event(
                FittingEventType.END_ATTEMPT , 'EarlyStop' , epoch = self.metric_best_epoch , 
                message = f'Global2Top combined accuracy converged at {epoch_key(self.metric_best_epoch, 1)}, recognized at {self.status.epoch_key}'
            )

    def recall_global_climax(self):
        if self.fitting_phase == 'global' and self.global_plateaued:
            self.trainer.recall_ckpt(
                self.global_climax_epoch , 0 , 
                message = f'Global2Top global accuracy plateaued at {epoch_key(self.global_climax_epoch, 0)} , recall ckpt at {self.status.epoch_key}'
            )
            self.fitting_phase = 'top'
    
    def set_loss_weights(self):
        # before plateau, set the weights of the top loss to 0
        if not self.is_fitting or self.loss_weights_set:
            return
        losses = self.metrics.batch_metrics.losses
        top_names = [name for name in losses.keys() if name.startswith('top.')]
        assert top_names , f'top losses should exist: {losses}'

        if self.global_plateaued:
            top_alpha = sigmoid_curve(self.status.epoch - self.global_climax_epoch , midpoint=10.0)
        else:
            top_alpha = 0.
        weights : dict[str,float] = {name:top_alpha for name in top_names}
        self.metrics.set_loss_weights(weights)
        self.loss_weights_set = True

    def set_accuracy_aggregator(self):
        if not self.is_fitting or self.accuracy_computer_set:
            return
        aggregator = aggregator_factory(self.fitting_phase , glb_climax = self.global_climax_level)
        self.metrics.set_accuracy_aggregator(aggregator)
        self.accuracy_computer_set = True

    def set_accuracy_verdict(self):
        aggregator = aggregator_factory(self.fitting_phase , glb_climax = None)
        self.metrics.set_accuracy_verdict(aggregator)

    def on_configure_model(self):
        self.override_configuration()
        
    def on_fit_model_start(self):
        self.reset_record()

    def on_batch_metrics_after(self):
        self.set_loss_weights()
        self.set_accuracy_aggregator()

    def on_fit_epoch_start(self):
        self.loss_weights_set = False
        self.accuracy_computer_set = False

    def on_validation_epoch_end(self):
        self.collect_accuracies()
        self.check_global_plateau()
        self.check_top_converge()
        self.recall_global_climax()

    
        
        