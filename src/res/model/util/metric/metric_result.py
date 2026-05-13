from __future__ import annotations
from dataclasses import dataclass
from typing import Any , Literal

from src.res.model.util.core import epoch_key

@dataclass
class EpochMetricResult:
    epoch : int
    phase : int
    valid_accuracies : dict[str,float]
    valid_losses : dict[str,float]
    valid_rankic : float
    valid_total_accuracy : float
    valid_total_loss : float
    train_accuracies : dict[str,float]
    train_losses : dict[str,float]
    train_rankic : float
    train_total_accuracy : float
    train_total_loss : float

    @property
    def epoch_key(self):
        return epoch_key(self.epoch , self.phase)

    def metrics(self , dataset : Literal['train','valid'] , metric : Literal['accuracy','loss']) -> dict[Any,float]:
        if metric == 'accuracy':
            return self.valid_accuracies if dataset == 'valid' else self.train_accuracies
        elif metric == 'loss':
            return self.valid_losses if dataset == 'valid' else self.train_losses
        else:
            raise ValueError(f'Invalid metric: {metric}')
