"""Opt-in model-level CUDA OOM recovery."""
from __future__ import annotations

import torch

from src.res.algo.nn.layer.checkpoint import ActivationCheckpointMixin, is_cuda_oom
from src.res.model.util import BaseCallBack

__all__ = ['ActivationCheckpointing']


class ActivationCheckpointing(BaseCallBack):
    """Automatically restart an OOM model with activation checkpointing.

    Configuration (schedule YAML)::

        callbacks:
          ActivationCheckpointing:
            enabled: true

    The callback is registered in train.callbacks by default but enabled=false.
    Explicit enabled=true also registers it for older/custom callback lists.
    Disabled means
    no validation or recovery. Enabled requires an ActivationCheckpointMixin
    model declaring nonempty, callable regions; invalid models fail before the
    first batch. The first model trains normally. The first CUDA OOM inside
    fit() discards ALL attempts of the current (model_date, model_num), including
    optimizer, best candidates, phase/early-stop state and temporary checkpoints.
    Training restarts at trial 0 / epoch 0 with ordinary new-model initialization
    (including the configured transfer policy), not a saved training node.
    Earlier completed models survive. All subsequent models and quality retries
    in this trainer run use checkpointing. A second CUDA OOM propagates. New
    processes/resume start in normal mode again. Setup/save/test errors are not
    retried. This is unrelated to disk checkpoint files used for training resume.

    To support a new model, inherit ActivationCheckpointMixin before nn.Module,
    declare pure region methods, and use checkpoint_region in forward. Example
    from transformer_gru (the GRU and BatchNorm output head remain unchanged)::

        class transformer_gru(ActivationCheckpointMixin, nn.Module):
            activation_checkpoint_regions = ('encode_day',)

            def encode_day(self, day):
                return self.fc_enc_in(day.contiguous())[:, -1]

            def forward(self, x):
                daily = torch.stack([
                    self.checkpoint_region('encode_day', day)
                    for day in x.unbind(1)
                ], dim=1)
                # Existing GRU, output head and loss consume daily unchanged.

    Import the mixin from src.res.algo.nn.layer.checkpoint. Regions must avoid
    mutable caches, counters and BatchNorm running-stat updates: backward reruns
    their forward computations. Keep such operations outside the regions.
    Dropout is supported via preserve_rng_state=True. use_reentrant=False allows
    ordinary input data without requires_grad. Eval/no-grad bypass checkpointing.
    No parameters/state_dict keys or precision settings change; full-batch losses
    and FP32 are preserved. Test identical initial weights, inputs and RNG state
    with dropout enabled for output/gradient equivalence and strict state_dict
    loading. Measure CUDA full forward/backward peak memory separately.

    Recomputing activations costs time. This does not guarantee enough memory:
    resident data, parameters, optimizer state and workspaces can still OOM.
    """
    CB_KEY_PARAMS = ['enabled']

    def __init__(self, trainer, enabled=False, **kwargs):
        super().__init__(trainer, **kwargs)
        if not isinstance(enabled, bool):
            raise ValueError('ActivationCheckpointing.enabled must be a boolean')
        self.enabled = enabled
        self.active = False

    def __bool__(self):
        return self.enabled

    def on_fit_start_before(self):
        if self.config.module_type != 'nn':
            raise ValueError('ActivationCheckpointing requires a neural network model')

    def on_new_attempt(self):
        net = self.model.persist_net()
        if not isinstance(net, ActivationCheckpointMixin):
            raise ValueError(f'{type(net).__name__} must explicitly declare activation checkpoint regions '
                             'using ActivationCheckpointMixin')
        net.set_activation_checkpointing(self.active)

    def on_fit_model_restart(self):
        # Deliberately survive the model-level callback reset.
        pass

    def request_fit_restart(self, exc: BaseException) -> bool:
        if not self.enabled or not is_cuda_oom(exc):
            return False
        def shapes(value):
            if isinstance(value, torch.Tensor):
                return tuple(value.shape)
            if isinstance(value, (list, tuple)):
                return [shapes(item) for item in value]
            if isinstance(value, dict):
                return {key: shapes(item) for key, item in value.items()}
            return None
        memory = {}
        try:
            device = next(self.model.persist_net().parameters()).device
            if device.type == 'cuda':
                memory = dict(allocated=torch.cuda.memory_allocated(device),
                              reserved=torch.cuda.memory_reserved(device),
                              peak=torch.cuda.max_memory_allocated(device))
        except (RuntimeError, StopIteration):
            pass
        detail = (f'model_date={self.model_date}, model_num={self.model_num}, '
                  f'attempt={self.status.attempt}, phase={self.status.phase}, '
                  f'epoch={self.status.epoch}, dataset={self.status.dataset}, '
                  f'batch={getattr(self.trainer, "batch_idx", None)}, '
                  f'shapes={shapes(getattr(getattr(self.trainer, "batch_input", None), "x", None))}, '
                  f'CUDA bytes={memory}; {exc}')
        if self.active:
            self.logger.warning(f'CUDA OOM with activation checkpointing already enabled; aborting. {detail}')
            return False
        self.active = True
        self.logger.warning('CUDA OOM: discarding ALL attempts of current model and restarting at '
                            f'trial 0 / epoch 0 with activation checkpointing. {detail}')
        return True
