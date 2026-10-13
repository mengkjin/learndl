"""Fixed linear combination of the already preprocessed final input columns."""
from __future__ import annotations

import torch
from src.res.model.util import PredictorModel, BatchInput


class LinearPredictor(PredictorModel):
    def init_model(self, *args, **kwargs):
        self.model_dict.reset()
        self.complete_model_param = {}
        return self

    def reload_model(self, *args, **kwargs):
        return self.init_model()

    def load_model(self, *args, **kwargs):
        # A schedule, not a trained checkpoint, defines this predictor.
        return self.init_model()

    def forward(self, batch_input: BatchInput | torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], *args, **kwargs):
        x = batch_input.x if isinstance(batch_input, BatchInput) else batch_input
        inputs: list[torch.Tensor] = [x] if isinstance(x, torch.Tensor) else list(x)
        columns = []
        for value in inputs:
            if value.ndim < 2 or any(size != 1 for size in value.shape[1:]):
                raise ValueError(f'Linear final inputs must be scalar per stock, got {value.shape}')
            columns.append(value.reshape(value.shape[0], 1))
        return torch.stack(columns, dim=0).sum(dim=0)

    def fit(self):
        raise RuntimeError('Fixed linear models have no fit stage')

    def ckpt_state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return self

    def collect(self, *args):
        return self.model_dict
