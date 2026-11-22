"""Checkpoint selection and stochastic weight averaging for the trainer."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from time import monotonic
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from src.res.model.util import BatchInput, Checkpoint, EpochMetricResult
from src.res.model.util.trainer.phase_transition import auxiliary_batch, preserve_module_modes, preserve_rng
from src.res.model.util.metric.aggregator import MetricAggregator
from src.res.model.util.core import BatchData, BatchOutput

SWAEnsemblerType: TypeAlias = Literal['best', 'swabest', 'swalast', 'swa']

__all__ = ['choose_swa_method', 'SWAEnsembler', 'SWAModel', 'EnsembleBestOne',
           'EnsembleSWABest', 'EnsembleSWALast', 'EnsembleSWA']


def choose_swa_method(submodel: str):
    return {'best': EnsembleBestOne, 'swabest': EnsembleSWABest,
            'swalast': EnsembleSWALast, 'swa': EnsembleSWA}[submodel]


class SWAEnsembler(ABC):
    def __init__(self, ckpt: Checkpoint, *args, **kwargs) -> None:
        self.ckpt = ckpt
        self.reset()

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def assess(self, status, metrics): ...

    @abstractmethod
    def collect(self, trainer, *args, **kwargs) -> nn.Module: ...

    def source_epochs(self) -> list[tuple[int, int]]:
        return []

    def average_checkpoints(self, trainer, epochs):
        if not epochs:
            raise ValueError('No checkpoints selected for SWA')
        swa = SWAModel(trainer.model.persist_net())
        for epoch, phase in epochs:
            swa.update_sd(self.ckpt.load(epoch, phase)['net'])
        return swa.update_bn(trainer).module.cpu().eval()


class SWAModel:
    def __init__(self, module: nn.Module) -> None:
        self.template = deepcopy(module).cpu()
        self.avgmodel = AveragedModel(self.template)

    def update_sd(self, state_dict):
        self.template.load_state_dict(state_dict)
        self.avgmodel.update_parameters(self.template)
        return self

    def update_bn(self, trainer):
        device = getattr(trainer.device, 'device', trainer.device)
        self.avgmodel.to(device)
        with preserve_rng():
            update_swa_bn(self.bn_loader(trainer), self.avgmodel)
        return self

    def bn_loader(self, trainer):
        for batch_input in trainer.data.train_dataloader():
            assert isinstance(batch_input, BatchInput)
            if len(batch_input):
                batch = auxiliary_batch(trainer, batch_input, training=True)
                yield batch.x, batch.kwargs

    @property
    def module(self) -> nn.Module:
        return self.avgmodel.module


@torch.no_grad()
def update_swa_bn(loader, model: nn.Module):
    momenta = {
        module: module.momentum for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats
    }
    if not momenta:
        return

    with preserve_module_modes(model):
        try:
            model.train()
            for module in momenta:
                module.reset_running_stats()
                module.momentum = None
            count = 0
            for x, kwargs in loader:
                model(x, **kwargs)
                count += 1
            if not count:
                raise ValueError('No training batches available for SWA BatchNorm calibration')
        finally:
            for module, momentum in momenta.items():
                module.momentum = momentum


class EnsembleBestOne(SWAEnsembler):
    """State dict of the best checkpoint under the existing epoch criterion."""
    def reset(self):
        self.best_epoch: EpochMetricResult | None = None

    def assess(self, status, metrics):
        latest = metrics.attempt_metrics.latest_epoch()
        if latest is not None and metrics.compare_epochs(latest, self.best_epoch):
            self.select_best(latest, metrics)

    def select_best(self, epoch, metrics):
        self.ckpt.join(self, epoch.epoch, epoch.phase)
        if self.best_epoch and self.source_epochs() != [(epoch.epoch, epoch.phase)]:
            self.ckpt.disjoin(self, self.best_epoch.epoch, self.best_epoch.phase)
        self.best_epoch = epoch

    def source_epochs(self):
        epoch = self.best_epoch
        return [(epoch.epoch, epoch.phase)] if epoch else []

    def collect(self, trainer, *args, **kwargs):
        net = deepcopy(trainer.model.persist_net())
        if self.best_epoch:
            net.load_state_dict(self.ckpt.load(self.best_epoch.epoch, self.best_epoch.phase)['net'])
        return net


class EnsembleSWABest(SWAEnsembler):
    """Average the checkpoints with the highest scores under the fixed model verdict."""
    def __init__(self, ckpt: Checkpoint, n_best=5, *args, **kwargs) -> None:
        if n_best < 1:
            raise ValueError('n_best must be positive')
        self.n_best = n_best
        super().__init__(ckpt, *args, **kwargs)

    def reset(self):
        self.top_epochs: list[EpochMetricResult] = []

    def assess(self, status, metrics):
        latest = metrics.attempt_metrics.latest_epoch()
        if latest is None or (latest.epoch, latest.phase) in self.source_epochs():
            return

        verdict = metrics.model_metrics.accuracy_verdict
        def score(epoch):
            return metrics.aggregator.total_accuracy(epoch.valid_accuracies, verdict if verdict is not None else {})

        latest_score = score(latest)
        if not math.isfinite(latest_score):
            return
        if len(self.top_epochs) == self.n_best:
            # On ties, retain earlier checkpoints.
            index = min(range(len(self.top_epochs)), key=lambda i: (score(self.top_epochs[i]), -i))
            if latest_score <= score(self.top_epochs[index]):
                return
            drop = self.top_epochs.pop(index)
            self.ckpt.disjoin(self, drop.epoch, drop.phase)
        self.ckpt.join(self, latest.epoch, latest.phase)
        self.top_epochs.append(latest)

    def source_epochs(self):
        return [(epoch.epoch, epoch.phase) for epoch in self.top_epochs]

    def collect(self, trainer, *args, **kwargs):
        return self.average_checkpoints(trainer, self.source_epochs())


class EnsembleSWALast(SWAEnsembler):
    """Average observed periodic checkpoints around the best point of the current phase."""
    def __init__(self, ckpt: Checkpoint, n_last=5, interval=3, *args, **kwargs) -> None:
        if n_last < 1 or interval < 1:
            raise ValueError('n_last and interval must be positive')
        self.n_last = n_last
        self.interval = interval
        super().__init__(ckpt, *args, **kwargs)

    def reset(self):
        self.best_epoch: EpochMetricResult | None = None
        self.periodic_epochs: list[int] = []
        self.adjacent_epochs: list[tuple[int, int]] = []
        self.retained: set[tuple[int, int]] = set()

    def assess(self, status, metrics):
        latest = metrics.attempt_metrics.latest_epoch()
        if latest is None:
            return
        if self.best_epoch is None or latest.phase != self.best_epoch.phase:
            self.periodic_epochs.clear()
            self.best_epoch = latest
        elif metrics.compare_epochs(latest, self.best_epoch):
            self.best_epoch = latest

        epoch, phase = latest.epoch, latest.phase
        if epoch >= self.interval and epoch % self.interval == 0 and epoch not in self.periodic_epochs:
            self.periodic_epochs.append(epoch)
        assert self.best_epoch is not None
        self.adjacent_epochs = self.interval_epochs(self.best_epoch.epoch, epoch, phase)
        # Keep the nearest past samples for a future best, as well as the current window.
        n_left = (self.n_last - 1) // 2
        recent = self.periodic_epochs[-n_left:] if n_left else []
        needed = set(self.source_epochs()) | {(ep, phase) for ep in recent}
        for ep, ph in self.retained - needed:
            self.ckpt.disjoin(self, ep, ph)
        for ep, ph in needed - self.retained:
            self.ckpt.join(self, ep, ph)
        self.retained = needed

    def interval_epochs(self, fix_epoch: int, epoch: int, phase: int = 0):
        left = [ep for ep in self.periodic_epochs if ep < fix_epoch]
        right = [ep for ep in self.periodic_epochs if fix_epoch < ep <= epoch]
        n_left = (self.n_last - 1) // 2
        left = left[-n_left:] if n_left else []
        return [(ep, phase) for ep in [*left, fix_epoch, *right][:self.n_last]]

    def source_epochs(self):
        return list(self.adjacent_epochs)

    def collect(self, trainer, *args, **kwargs):
        return self.average_checkpoints(trainer, self.source_epochs())


class EnsembleSWA(EnsembleBestOne):
    """Continue from this attempt's best checkpoint on a fixed local objective."""
    def reset(self):
        super().reset()
        self.loss_function = None
        self.loss_aggregator = None
        self.continuation = {}

    def select_best(self, epoch, metrics):
        previous = self.best_epoch
        super().select_best(epoch, metrics)
        if self.best_epoch is not previous:
            from src.res.model.util.metric.aggregator import LossAggregator

            net = metrics.loss_function.net
            memo = {id(net): net, **{id(p): p for p in net.parameters()}} if net is not None else {}
            self.loss_function = deepcopy(metrics.loss_function, memo)
            state = self.ckpt.load(epoch.epoch, epoch.phase)
            self.loss_aggregator = LossAggregator(deepcopy(state['swa_loss_weights']))

    def collect(self, trainer, *args, **kwargs):
        best = self.best_epoch
        if best is None or self.loss_function is None or self.loss_aggregator is None:
            raise ValueError('No best checkpoint with a frozen objective for SWA')
        epochs, lr_ratio = trainer.config.swa_settings
        state = self.ckpt.load(best.epoch, best.phase)
        started = monotonic()
        with preserve_rng():
            swa, trace = self.continue_fit(trainer, state, epochs, lr_ratio)
            net = swa.update_bn(trainer).module.cpu().eval()
        elapsed = monotonic() - started
        self.continuation = {'swa_epochs': epochs, 'swa_lr_ratio': lr_ratio, 'swa_seconds': elapsed,
                             'swa_trace': trace,
                             'swa_group_lrs': [group['lr_param'] * lr_ratio for group in state['optimizer']['param_groups']]}
        trainer.write_status_message(f'SWA complete: {epochs} epochs in {elapsed:.1f}s')
        return net

    @torch.enable_grad()
    def continue_fit(self, trainer, state, epochs, lr_ratio):
        from src.res.model.util.metric.functions import NonFiniteAccuracyError
        from src.res.model.util.metric.trainer_metrics import copy_metric_function

        predictor = trainer.model
        original = predictor.persist_net()
        device = getattr(trainer.device, 'device', trainer.device)
        net = deepcopy(original).to(device).train()
        net.load_state_dict(state['net'])
        assert self.loss_function is not None and self.loss_aggregator is not None
        loss_function = copy_metric_function(self.loss_function, net)
        loss_weights = self.loss_aggregator.aggregator
        aggregator = MetricAggregator()
        aggregator.loss_aggregators[-1] = deepcopy(self.loss_aggregator)

        # Recreate the parameter groups without rebuilding the scheduler or initializing weights.
        candidate_params = dict(net.named_parameters())
        optimizer = copy(predictor.optimizer)
        live = optimizer.optimizer
        groups = [{'params': [candidate_params[name] for name in names]}
                  for names in state['optimizer_param_names']]
        optimizer.optimizer = type(live)(groups)
        optimizer.optimizer.load_state_dict(deepcopy(state['optimizer']))
        for group in optimizer.optimizer.param_groups:
            group['lr'] = group['lr_param'] * lr_ratio
        optimizer.net = net
        optimizer.trainer = None

        def forward(batch):
            return net(batch.x, **batch.kwargs)

        def validation_ic():
            try:
                metrics = trainer.metrics.evaluate_split(
                    net, forward, 'valid', loss_function=loss_function,
                    loss_weights=loss_weights,
                )
            except NonFiniteAccuracyError as exc:
                trainer.write_status_message(f'SWA validation unavailable: {exc}')
                return float('nan')
            return float(metrics['valid_rankic'])

        anchor_ic = validation_ic()
        trainer.write_status_message(f'SWA anchor {state["epoch"]} phase {state["phase"]} ValidIC={anchor_ic:.6f}')
        trace = [{'step': 0, 'valid_rankic': anchor_ic}]
        swa = SWAModel(original)
        for step in range(1, epochs + 1):
            count = 0
            total_loss = 0.
            for batch in trainer.data.train_dataloader():
                batch = auxiliary_batch(trainer, batch, training=True)
                if not len(batch):
                    continue
                data = BatchData(batch, BatchOutput(forward(batch)))
                if data.output.empty:
                    continue
                losses = loss_function.losses(data, trainer.metrics.which_output, trainer.metrics.which_label)
                loss = aggregator.total_loss(losses)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss during SWA continuation')
                optimizer.optimizer.zero_grad()
                loss.backward()
                optimizer.clip_gradients()
                optimizer.optimizer.step()
                total_loss += loss.detach().item()
                count += 1
            if not count:
                raise ValueError('No training batches available for SWA continuation')
            swa.avgmodel.update_parameters(net)
            loss_mean = total_loss / count
            valid_ic = validation_ic()
            trace.append({'step': step, 'train_loss': loss_mean, 'valid_rankic': valid_ic})
            trainer.write_status_message(f'SWA epoch {step}/{epochs} TrainLoss={loss_mean:.6f} ValidIC={valid_ic:.6f}')
        return swa, trace
