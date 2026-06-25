"""
Callback for MOE models: log per-expert mean gate weights on validation epochs.
"""
from __future__ import annotations

import torch
from functools import cached_property
from torch.utils.tensorboard import SummaryWriter as TsboardWriter

from src.res.model.util import BaseCallBack

__all__ = ['SpecificCB_MOE']


class SpecificCB_MOE(BaseCallBack):
    """Accumulate validation gate weights and log expert usage to tensorboard."""

    CB_KEY_PARAMS : list[str] = []

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.gate_sum : torch.Tensor | None = None
        self.gate_count : int = 0

    @cached_property
    def writer(self) -> TsboardWriter:
        model_key = (
            f'{self.config.base_path.model_clean_name}.'
            f'{self.model_num}.{self.model_date}.{self.status.attempt_key}'
        )
        return TsboardWriter(self.config.base_path.snapshot('tensorboard' , model_key))

    def reset_gate_record(self) -> None:
        self.gate_sum = None
        self.gate_count = 0

    def accumulate_gate(self) -> None:
        if self.status.dataset != 'valid':
            return
        gate = self.batch_output.other.get('gate')
        if gate is None or gate.numel() == 0:
            return
        batch_mean = gate.detach().mean(dim = 0).cpu()
        if self.gate_sum is None:
            self.gate_sum = batch_mean.clone()
        else:
            self.gate_sum += batch_mean
        self.gate_count += 1

    def log_gate_means(self) -> None:
        if self.gate_sum is None or self.gate_count == 0:
            return
        gate_mean = (self.gate_sum / self.gate_count).tolist()
        msg = ', '.join([f'expert_{i}={w:.4f}' for i , w in enumerate(gate_mean)])
        self.logger.note(f'MOE gate mean (valid epoch {self.status.epoch}): {msg}' , vb_level = 'max')
        step = self.status.epoch
        for i , weight in enumerate(gate_mean):
            self.writer.add_scalar(f'02.HiddenFeatures/MOE/gate_expert_{i}' , weight , step)

    def on_fit_model_start(self) -> None:
        self.reset_gate_record()

    def on_batch_metrics_after(self) -> None:
        self.accumulate_gate()

    def on_validation_epoch_end(self) -> None:
        self.log_gate_means()
        self.reset_gate_record()
