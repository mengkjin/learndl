"""Optional ``torch.compile`` integration for NN predictors via :class:`TorchCompiler`."""
from __future__ import annotations


import torch
from datetime import datetime
from torch import nn
from typing import TYPE_CHECKING , Any , Literal , cast

from src.proj import MACHINE , Base

if TYPE_CHECKING:
    from src.proj.bases.classes.bound_logger import ModuleLogger
    from src.res.model.util.trainer import PredictorModel
    from src.res.model.util.config import ModelConfig

__all__ = ['CompileStage', 'TorchCompiler']

CompileStage = Literal['fit', 'test']

_COMPILED_MARK = '_learndl_torch_compiled'

class TorchCompiler:
    """Manage torch.compile wrap, warmup timing, and eager fallback for one predictor."""

    def __init__(self, predictor: PredictorModel) -> None:
        self._disabled = not MACHINE.cuda_server
        self._predictor = predictor

        self._raw: nn.Module | None = None
        self._active: nn.Module | None = None
        self._warmup_logged = False

    @classmethod
    def is_compiled(cls, module: nn.Module) -> bool:
        """Return True if *module* was wrapped by this helper."""
        return bool(getattr(module, _COMPILED_MARK, False))

    @classmethod
    def set_compiled(cls, module: nn.Module) -> None:
        """Set the compiled mark to True."""
        setattr(module, _COMPILED_MARK, True)

    @classmethod
    def reset_compiled(cls, module: nn.Module) -> None:
        """Set the compiled mark to False."""
        if cls.is_compiled(module):
            setattr(module, _COMPILED_MARK, False)

    @classmethod
    def unwrap_module(cls, module: nn.Module) -> nn.Module:
        """PyTorch-only unwrap via ``_orig_mod`` when no :class:`TorchCompiler` context."""
        if cls.is_compiled(module):
            return getattr(module, '_orig_mod', module)
        orig = getattr(module, '_orig_mod', None)
        return orig if isinstance(orig, nn.Module) else module

    @property
    def raw(self) -> nn.Module | None:
        """The unwrapped module passed to :meth:`wrap` (authoritative when set)."""
        return self._raw

    def unwrap(self, module: nn.Module | None = None) -> nn.Module:
        """Return the eager ``nn.Module``, preferring :attr:`raw` over PyTorch ``_orig_mod``."""
        mod = self.net if module is None else module
        if mod is None:
            raise RuntimeError('TorchCompiler.unwrap called with no module')
        if self._raw is not None and (mod is self._raw or mod is self._active):
            return self._raw
        return self.unwrap_module(mod)

    @classmethod
    def is_compile_error(cls, exc: BaseException) -> bool:
        """True when *exc* originates from torch.compile / Dynamo / Inductor."""
        inductor_error = getattr(getattr(torch, '_inductor', None), 'exc', None)
        if inductor_error is not None and isinstance(exc, getattr(inductor_error, 'InductorError', ())):
            return True
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            module_name = type(current).__module__
            type_name = type(current).__name__
            if module_name.startswith(('torch._dynamo', 'torch._inductor')):
                return True
            if type_name in {'InductorError', 'BackendCompilerFailed', 'LoweringException'}:
                return True
            current = current.__cause__ or current.__context__
        return False

    @classmethod
    def stage_enabled(cls, config_stage: str, current_stage: CompileStage | None) -> bool:
        stage = (config_stage or 'fit').lower().replace(' ', '')
        if stage in ('both', 'all', 'fit+test'):
            return True
        if current_stage is None:
            return stage == 'fit'
        return stage == current_stage

    @property
    def eligible(self) -> bool:
        return self.model.AllowTorchCompile

    @property
    def enabled(self) -> bool:
        if not self.eligible or self._disabled:
            return False
        config = self._predictor.config
        if not config or not bool(config['train.torch_compile']):
            return False
        if not hasattr(torch, 'compile'):
            return False
        return self.stage_enabled(
            str(config['train.torch_compile.stage']),
            self.stage,
        )

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def logger(self) -> ModuleLogger:
        return self._predictor.logger

    @property
    def model(self) -> PredictorModel:
        return self._predictor

    @property
    def net(self) -> nn.Module | None:
        return self._predictor.net

    @property
    def config(self) -> ModelConfig:
        return self._predictor.config

    @property
    def stage(self) -> CompileStage:
        if self._predictor.bounded_with_trainer:
            stage = self._predictor.trainer.status.stage
            assert stage == 'fit' or stage == 'test', f'Invalid stage: {stage}'
            return stage
        return 'fit'

    def reset(self) -> None:
        """Clear per-model lifecycle state before wrapping a fresh ``nn.Module``."""
        self._raw = None
        self._active = None
        self._warmup_logged = False
        if self.net is not None:
            self.reset_compiled(self.net)

    def wrap(self, module: nn.Module) -> nn.Module:
        """Optionally wrap *module* with ``torch.compile``; always store raw/active refs."""
        assert self.eligible , f'TorchCompiler is not eligible, predictor.AllowTorchCompile is {self._predictor.AllowTorchCompile} , net is {self._predictor.net is not None}'
        self._raw = module
        self._active = module
        if not self.enabled:
            return module
        t0 = datetime.now()
        compile_kwargs: dict[str, Any] = {
            'mode': str(self.config['train.torch_compile.mode']),
            'fullgraph': bool(self.config['train.torch_compile.fullgraph']),
            'dynamic': True,
        }
        compiled = cast(nn.Module, torch.compile(module, **compile_kwargs))
        self.set_compiled(compiled)
        self._active = compiled
        
        self.logger.note(
            f'torch.compile enabled (stage={self.stage}, '
            f'mode={compile_kwargs["mode"]}, dynamic=True, fullgraph={compile_kwargs["fullgraph"]})'
            f'elapsed: {Base.Since(t0)}'
        )
        return compiled

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Forward through compiled module with eager fallback on compile errors."""
        if self._raw is None or self._active is None:
            raise RuntimeError('TorchCompiler.run called before wrap()')
        if self._disabled or self._active is self._raw:
            return self._raw(*args, **kwargs)

        t0 = datetime.now()
        try:
            output = self._active(*args, **kwargs)
        except Exception as exc:
            if not self.is_compile_error(exc):
                raise
            self.logger.alert2(
                f'torch.compile set to disabled due to compile error, falling back to eager mode for this model'
                f'Failed after {Base.Since(t0)} ({type(exc).__name__}: {exc}); '
            )
            self.logger.print_exc(exc)
            self._fallback()
            return self._raw(*args, **kwargs)

        if not self._warmup_logged:
            self.logger.note(f'torch.compile warmup finished in {Base.Since(t0)}')
            self._warmup_logged = True
        return output

    def _fallback(self) -> None:
        assert self._raw is not None
        self._disabled = True
        self._active = self._raw
        self.model.net = self._raw

