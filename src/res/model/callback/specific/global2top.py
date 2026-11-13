"""
Callback specifically for Global2Top module
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from src.res.model.util.trainer.phase_transition import validate_recalled_model, capture_cycle, PhaseCyclicLR

from typing import Literal , TypeAlias
from collections.abc import Callable
from src.proj.bases import FittingEventType
from src.res.model.util import BaseCallBack
from src.res.model.util.core import epoch_key

__all__ = ['SpecificCB_Global2Top']

G2TPhase : TypeAlias = Literal['global' , 'top']

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

def aggregator_factory(phase : G2TPhase , glb_climax : float | None = None , glb_multiplier : float = 5.) -> Callable[[dict[str,float]],float]:
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
    CB_KEY_PARAMS = ['protect' , 'plateau_patience' , 'converge_patience' , 'top_converge_alpha', 'min_top_epochs', 'transition_epochs', 'verify_recall']
    OverrideCallbacks = ['EarlyStoppage']
    ConflictModuleTypes = ['factor']
    def __init__(self , trainer , 
        protect : int = 10 , 
        plateau_patience : int = 20 , 
        converge_patience : int = 10 ,
        top_converge_alpha : float = 0.6, min_top_epochs: int = 20,
        transition_epochs: int = 10, verify_recall: bool = True, gradient_diagnostics: bool = False, **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        assert 0 <= top_converge_alpha <= 1 , f'top_converge_alpha must be between 0 and 1 and is {top_converge_alpha}'
        if min_top_epochs < 1 or transition_epochs < 2 or converge_patience < 1:
            raise ValueError("Invalid Global2Top phase lengths")
        self.min_top_epochs = min_top_epochs
        self.transition_epochs = transition_epochs
        self.verify_recall = verify_recall
        self.gradient_diagnostics = gradient_diagnostics
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
        self.fitting_phase : G2TPhase = 'global'
        self.top_epochs = 0
        self.top_alpha = 0.
        self.top_best = -np.inf
        self.top_stale = 0
        self.top_low = 0
        self.pinned_epochs = []
        self.diagnostics = {}

    def collect_accuracies(self):
        accuracies = pd.concat(self.metrics.attempt_metrics.tables['valid_epoch_accuracies'])
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
            self.global_support_level = min(0.1, max(0 , min(self.global_climax_level * 0.9 , self.global_climax_level - 0.01)))

    def check_top_converge(self):
        if self.fitting_phase != 'top':
            return
        phase1 = self.valid_accuracies.query('phase == 1')
        if phase1.empty:
            return
        self.top_epochs = len(phase1)
        glb = phase1[self.glb_accus].sum(axis=1).to_numpy()
        top = phase1[self.top_accus].sum(axis=1).to_numpy()
        scores = synthetic_accuracy(glb, top, glb_climax=self.global_climax_level)
        best = int(np.argmax(scores))
        self.top_best = float(scores[best])
        self.top_stale = len(scores) - best - 1
        self.top_low = 0
        for value in glb[::-1]:
            if value >= self.global_support_level:
                break
            self.top_low += 1
        if self.top_epochs < self.min_top_epochs:
            return
        if self.top_stale >= self.converge_patience or self.top_low >= self.converge_patience:
            scores_all = synthetic_accuracy(self.global_accuracies, self.top_accuracies,
                                            glb_climax=self.global_climax_level)
            best_position = int(np.argmax(scores_all))
            index = self.valid_accuracies.index
            phase = int(index.get_level_values('phase')[best_position])
            epoch = int(index.get_level_values('epoch')[best_position])
            self.status.add_epoch_event(
                FittingEventType.END_ATTEMPT, 'EarlyStop', epoch=int(epoch),
                message=f'Global2Top converged after {self.top_epochs} phase1 epochs; '
                        f'best {epoch_key(int(epoch), int(phase))}, recognized at {self.status.epoch_key}')

    def recall_global_climax(self):
        if self.fitting_phase == 'global' and self.global_plateaued:
            cycle = capture_cycle(self.model.optimizer)
            expected = self.metrics.attempt_metrics.best_epoch()
            self.trainer.recall_ckpt(
                self.global_climax_epoch , 0 , 
                message = f'Global2Top global accuracy plateaued at {epoch_key(self.global_climax_epoch, 0)} , recall ckpt at {self.status.epoch_key}'
            )
            if self.verify_recall:
                actual_acc, actual_loss = validate_recalled_model(self.trainer)
                if expected is None:
                    self.logger.warning('Global2Top recall baseline unverified: missing saved metrics')
                else:
                    for kind, actual, reference in (
                        ('accuracy', actual_acc, expected.valid_accuracies),
                        ('loss', actual_loss, expected.valid_losses),
                    ):
                        if actual.keys() != reference.keys() or any(
                            not np.isclose(actual[k], reference[k], atol=1e-4, rtol=1e-3)
                            for k in reference
                        ):
                            raise RuntimeError(f'Global2Top recall baseline mismatch ({kind}): '
                                               f'expected={reference}, actual={actual}')
                    self.logger.note(f'Global2Top recall baseline verified: {actual_acc}; losses={actual_loss}')
            if cycle is not None:
                self.model.optimizer.scheduler = PhaseCyclicLR(
                    self.model.optimizer.optimizer, **cycle, warmup=self.transition_epochs)
            self.fitting_phase = 'top'
            # Reassess preserved phase0 candidates using exactly the phase1 score.
            aggregator = aggregator_factory('top', glb_climax=self.global_climax_level)
            self.metrics.aggregator.inject('accuracy', aggregator)
            submodel = getattr(self.model, 'submodels', {}).get('best')
            records = self.metrics.attempt_metrics.epoch_metric_results
            if submodel is not None and records:
                best = max(records, key=lambda row: aggregator(row.valid_accuracies))
                old = submodel.best_epoch
                self.checkpoint.join(submodel, best.epoch, best.phase)
                submodel.best_epoch = best
                if old is not None and (old.epoch, old.phase) != (best.epoch, best.phase):
                    self.checkpoint.disjoin(submodel, old.epoch, old.phase)
            for ep, ph in self.pinned_epochs:
                self.checkpoint.disjoin(self, ep, ph)
            self.pinned_epochs.clear()
    
    def set_loss_weights(self):
        # before plateau, set the weights of the top loss to 0
        if not self.is_fitting or self.loss_weights_set:
            return
        losses = self.metrics.batch_metrics.losses
        top_names = [name for name in losses.keys() if name.startswith('top.')]
        assert top_names , f'top losses should exist: {losses}'

        if self.global_plateaued:
            top_alpha = (1 - np.cos(np.pi * min(self.top_epochs / (self.transition_epochs - 1), 1.))) / 2
        else:
            top_alpha = 0.
        self.top_alpha = float(top_alpha)
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
        self.set_accuracy_verdict()

    def on_new_attempt(self):
        # Both quality retries and NaN redos start a fresh network. Keep the
        # model-level attempt verdict/history, but restart global -> top control.
        self.reset_record()

    def on_batch_metrics_after(self):
        self.set_loss_weights()
        self.set_accuracy_aggregator()

    def on_batch_backward_before(self):
        if (not self.gradient_diagnostics or self.fitting_phase != 'top'
                or self.top_epochs >= self.transition_epochs or self.trainer.batch_idx != 0):
            return
        losses = self.metrics.batch_metrics.losses
        parameters = [p for p in self.model.net.parameters() if p.requires_grad]
        groups = []
        for prefix in ('global.', 'top.'):
            loss = torch.stack([value.sum() for key, value in losses.items() if key.startswith(prefix)]).sum()
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            groups.append([torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)])
        norms = [torch.stack([g.square().sum() for g in group]).sum().sqrt() for group in groups]
        dot = torch.stack([(g * h).sum() for g, h in zip(*groups)]).sum()
        self.diagnostics = {'GlobalGradNorm': norms[0].item(), 'TopGradNorm': norms[1].item(),
                            'GradCosine': (dot / (norms[0] * norms[1]).clamp_min(1e-12)).item()}

    def on_fit_epoch_start(self):
        self.diagnostics = {}
        self.loss_weights_set = False
        self.accuracy_computer_set = False

    def on_validation_epoch_end(self):
        if self.fitting_phase == 'global':
            self.checkpoint.join(self, self.status.epoch, 0)
            self.pinned_epochs.append((self.status.epoch, 0))
        self.collect_accuracies()
        self.check_global_plateau()
        self.check_top_converge()
        self.recall_global_climax()
