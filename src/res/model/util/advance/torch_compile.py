"""Optional ``torch.compile`` integration for NN predictors.

Lifecycle (answers common questions)
------------------------------------
1. **Fit → test transition** — No explicit "uncompile" step is required. Each
   ``reload_model()`` / ``load_model()`` calls ``init_model()`` and builds a
   fresh ``self.net``. The previous compiled wrapper is dropped with the old
   reference. Configure ``train.torch_compile.stage`` to control which stages
   compile (default: ``fit`` only).

2. **Mutable values used in ``forward``** —
   * ``nn.Parameter`` / ``register_buffer`` — safe; updated in-place, traced as
     graph inputs.
   * Plain Python ``float`` / ``int`` on ``self`` changed between steps — may
     cause **guard failures** and recompilation, or be **specialized as
     constants** at first compile (stale value bug). Prefer ``nn.Parameter``,
     ``register_buffer``, or pass values as forward arguments.
   * Custom setters that replace tensors — treat like structure change; may
     require a new module instance / recompile.

3. **Dict returns and graph breaks** — A *graph break* means TorchDynamo stops
   tracing and runs the rest in eager Python. Returning
   ``{'hidden': tensor, ...}`` where **all values are tensors produced inside
   the traced region** usually keeps the heavy compute compiled; only dict
   construction / tuple unpacking at the boundary may graph-break with small
   overhead. Breaks are costly when values are **non-tensors**, keys are
   dynamic, or tensors are built outside the traced subgraph.
"""
from __future__ import annotations

import torch
from torch import nn
from typing import Any , Literal , cast

from src.proj import Logger
from src.res.model.util.config import ModelConfig

__all__ = [
    'CompileStage',
    'apply_torch_compile',
    'unwrap_compiled_module',
    'is_compiled_module',
]

CompileStage = Literal['fit', 'test']

_COMPILED_MARK = '_learndl_torch_compiled'


def is_compiled_module(module: nn.Module) -> bool:
    """Return True if *module* was wrapped by :func:`apply_torch_compile`."""
    return bool(getattr(module, _COMPILED_MARK, False))


def unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the inner ``nn.Module`` if *module* is torch-compiled, else *module*."""
    if is_compiled_module(module):
        return getattr(module, '_orig_mod', module)
    # PyTorch >= 2.0 may wrap without our mark when compile is applied externally.
    orig = getattr(module, '_orig_mod', None)
    return orig if isinstance(orig, nn.Module) else module


def _stage_enabled(config_stage: str, current_stage: CompileStage | None) -> bool:
    stage = (config_stage or 'fit').lower().replace(' ', '')
    if stage in ('both', 'all', 'fit+test'):
        return True
    if current_stage is None:
        return stage == 'fit'
    return stage == current_stage


def apply_torch_compile(
    module: nn.Module,
    *,
    enabled: bool,
    mode: str = 'default',
    dynamic: Literal[True] = True,
    fullgraph: bool = False,
    config_stage: str = 'fit',
    current_stage: CompileStage | None = None,
    logger: Logger | Any | None = None,
) -> nn.Module:
    """Wrap *module* with ``torch.compile`` when *enabled* and stage matches.

    Args:
        module:         Fresh ``nn.Module`` (already on target device).
        enabled:        Master switch (``train.torch_compile``).
        mode:           ``torch.compile`` mode (``default``, ``reduce-overhead``,
                        ``max-autotune``, …).
        dynamic:        Pass ``dynamic=`` to ``torch.compile`` (recommended for
                        varying daily stock counts).
        fullgraph:      If True, error on graph breaks instead of falling back.
        config_stage:   ``fit``, ``test``, or ``both`` / ``all``.
        current_stage:  Active trainer stage when the module is created.
        logger:         Optional logger for one-line compile notice.
    """
    if not enabled or not _stage_enabled(config_stage, current_stage):
        return module
    if not hasattr(torch, 'compile'):
        (logger or Logger).warning('train.torch_compile is enabled but torch.compile is unavailable')
        return module

    compile_kwargs: dict[str, Any] = {'mode': mode, 'fullgraph': fullgraph}
    assert dynamic is True , 'dynamic must be True in this project'
    compile_kwargs['dynamic'] = True

    compiled = cast(nn.Module, torch.compile(module, **compile_kwargs))
    setattr(compiled, _COMPILED_MARK, True)

    log = logger or Logger
    log.info(
        f'torch.compile applied (stage={current_stage or "fit"}, mode={mode}, '
        f'dynamic={dynamic}, fullgraph={fullgraph})'
    )
    return compiled


def apply_torch_compile_from_config(
    module: nn.Module,
    config: ModelConfig,
    current_stage: CompileStage | None,
    *,
    logger: Logger | Any | None = None,
) -> nn.Module:
    """Convenience wrapper reading ``train.torch_compile*`` keys from *config*."""
    return apply_torch_compile(
        module,
        enabled=bool(config['train.torch_compile']),
        mode=str(config['train.torch_compile.mode']),
        dynamic=True,
        fullgraph=bool(config['train.torch_compile.fullgraph']),
        config_stage=str(config['train.torch_compile.stage']),
        current_stage=current_stage,
        logger=logger,
    )
