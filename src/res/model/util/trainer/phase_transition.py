"""Isolated checkpoint validation and a serializable phase-local LR schedule."""
from __future__ import annotations

import math
import random
from typing import Any
from copy import copy, deepcopy

import numpy as np
import torch

__all__ = ['PhaseCyclicLR']

def validate_recalled_model(trainer):
    """Evaluate raw metrics without training hooks, histories, or scheduler steps."""
    metrics = copy(trainer.metrics)
    metrics.stage_metrics = {}
    metrics.aggregator = deepcopy(metrics.aggregator)
    metrics.accuracy_function = copy(metrics.accuracy_function)
    metrics.epoch_valid_metrics.new('valid')
    net = trainer.model.net
    modes = [(module, module.training) for module in net.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(), torch.no_grad():
            net.eval()
            for index, batch in enumerate(trainer.data.val_dataloader()):
                if index < trainer.batch_warm_up or index >= trainer.batch_aftermath:
                    continue
                data = trainer.model.get_batch_data(batch)
                if data.output.empty:
                    continue
                metrics.calculate('valid', index, data)
                # Materialize loss before appending (same as normal batch logging).
                _ = metrics.batch_metrics.total_loss
                metrics.epoch_valid_metrics.append(metrics.batch_metrics)
                metrics.batch_metrics.close()
            return (metrics.epoch_valid_metrics.accuracies_dict,
                    metrics.epoch_valid_metrics.losses_dict)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, training in modes:
            module.training = training


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
