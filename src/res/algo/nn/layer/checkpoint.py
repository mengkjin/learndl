"""Explicit, state-dict-neutral activation checkpoint regions for NN models."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def is_cuda_oom(exc: BaseException) -> bool:
    """Recognize CUDA OOM, including compiler wrappers, without swallowing CPU OOM."""
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, torch.cuda.OutOfMemoryError) and 'cpu' not in message:
            return True
        if isinstance(current, RuntimeError) and 'cuda' in message and 'out of memory' in message:
            return True
        current = current.__cause__ or current.__context__
    return False


class ActivationCheckpointMixin:
    """Declare pure methods in ``activation_checkpoint_regions`` and call via the helper."""
    training: bool  # provided by nn.Module
    activation_checkpoint_regions: tuple[str, ...] = ()
    activation_checkpointing_enabled = False

    def set_activation_checkpointing(self, enabled: bool) -> None:
        regions = self.activation_checkpoint_regions
        if (not isinstance(regions, (tuple, list)) or not regions
                or any(not isinstance(name, str) or not callable(getattr(self, name, None))
                       for name in regions)):
            raise ValueError(f'{type(self).__name__} must declare callable activation_checkpoint_regions')
        self.activation_checkpointing_enabled = enabled

    def checkpoint_region(self, name: str, *args, **kwargs) -> Any:
        """Return the named method's result unchanged, including nested tensor structures."""
        if name not in self.activation_checkpoint_regions:
            raise ValueError(f'Undeclared activation checkpoint region: {name}')
        function = getattr(self, name)
        if self.activation_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return function(*args, **kwargs)
