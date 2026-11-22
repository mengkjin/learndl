"""Isolated checkpoint validation and a serializable phase-local LR schedule."""
from __future__ import annotations

import math
import random
from typing import Any
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import replace

import numpy as np
import torch

__all__ = ['PhaseCyclicLR']

@contextmanager
def preserve_rng():
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=None if torch.cuda.is_initialized() else []):
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if mps is not None:
            torch.mps.set_rng_state(mps)


@contextmanager
def preserve_module_modes(net):
    modes = [(module, module.training) for module in net.modules()] if net is not None else []
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def auxiliary_batch(trainer, batch, *, training):
    """Prepare model kwargs without calling Trainer hooks or mutating its current batch."""
    if not len(batch):
        return batch
    kwargs = dict(batch.kwargs)
    for callback in trainer.callback.callbacks:
        prepare = getattr(callback, 'auxiliary_batch_kwargs', None)
        if prepare is not None:
            kwargs.update(prepare(batch, training=training))
    return replace(batch, kwargs=kwargs)


def evaluate_batches(trainer, metrics, batches, forward, *, dataset, epoch_metrics):
    epoch_metrics.new(dataset)
    for index, batch in enumerate(batches):
        if index < trainer.batch_warm_up or index >= trainer.batch_aftermath:
            continue
        data = forward(batch)
        if data.output.empty:
            continue
        # Evaluate the fit objective on both splits, without building gradients.
        metrics.calculate('valid', index, data)
        _ = metrics.batch_metrics.total_loss
        epoch_metrics.append(metrics.batch_metrics)
        metrics.batch_metrics.close()
    return epoch_metrics


def validate_recalled_model(trainer):
    """Evaluate raw metrics without training hooks, histories, or scheduler steps."""
    metrics = copy(trainer.metrics)
    metrics.stage_metrics = {}
    metrics.aggregator = deepcopy(metrics.aggregator)
    metrics.accuracy_function = copy(metrics.accuracy_function)
    net = trainer.model.net
    with preserve_rng(), preserve_module_modes(net), torch.no_grad():
        net.eval()
        epoch = evaluate_batches(trainer, metrics, trainer.data.val_dataloader(),
                                 trainer.model.get_batch_data, dataset='valid',
                                 epoch_metrics=metrics.epoch_valid_metrics)
        return epoch.accuracies_dict, epoch.losses_dict


class PhaseCyclicLR(torch.optim.lr_scheduler.LRScheduler):
    """Cosine warmup to the current decayed peak, then triangular2 cycles."""
    def __init__(self, optimizer, bases, peaks, starts, up, down, warmup=10):
        self.floors = list(bases)
        self.peaks = list(peaks)
        self.starts = list(starts)
        self.up, self.down, self.warmup = up, down, warmup
        super().__init__(optimizer)

    def get_lr(self):
        t = self.last_epoch
        if t < self.warmup:
            alpha = (1 - math.cos(math.pi * t / max(1, self.warmup - 1))) / 2
            return [s + (p - s) * alpha for s, p in zip(self.starts, self.peaks)]
        t -= self.warmup - 1
        if t <= self.down:
            factor = 1 - t / self.down
        else:
            t -= self.down
            cycle, pos = divmod(t, self.up + self.down)
            triangle = pos / self.up if pos <= self.up else 1 - (pos - self.up) / self.down
            factor = triangle / 2 ** (cycle + 1)
        return [b + (p - b) * factor for b, p in zip(self.floors, self.peaks)]


def capture_cycle(optimizer) -> dict[str, Any] | None:
    scheduler = optimizer.scheduler
    if not isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
        return None
    cycle = math.floor(1 + scheduler.last_epoch / scheduler.total_size)
    peaks = [b + (p - b) / 2 ** (cycle - 1)
             for b, p in zip(scheduler.base_lrs, scheduler.max_lrs)]
    return dict(bases=list(scheduler.base_lrs), peaks=peaks,
                starts=[max(b, min(pg['lr'], .1 * p)) for b, p, pg in
                        zip(scheduler.base_lrs, peaks, optimizer.optimizer.param_groups)],
                up=scheduler.total_size * scheduler.step_ratio,
                down=scheduler.total_size * (1 - scheduler.step_ratio))
