"""Script parameter schema: YAML header merged with ``main`` signature defaults."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.api.util.backend import ScriptHeader, ScriptParamInput

__all__ = ['ScriptParamSchema']


@dataclass
class ScriptParamSchema:
    """Ordered script parameters with merged default values."""

    script_path: Path
    params: list[ScriptParamInput]
    signature_defaults: dict[str, Any]

    @classmethod
    def from_script(cls, path: Path, main: Callable[..., Any] | None = None) -> ScriptParamSchema:
        """Build schema from script YAML header and optional ``main`` callable."""
        header = ScriptHeader.read_from_file(path)
        params = header.get_param_inputs()
        signature_defaults: dict[str, Any] = {}
        if main is not None:
            for name, parameter in inspect.signature(main).parameters.items():
                if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue
                if parameter.default is not inspect.Parameter.empty:
                    signature_defaults[name] = parameter.default
        return cls(script_path=path, params=params, signature_defaults=signature_defaults)

    def sig_default(self, name: str) -> Any:
        """Return signature default for *name*, or ``inspect.Parameter.empty``."""
        if name in self.signature_defaults:
            return self.signature_defaults[name]
        return inspect.Parameter.empty
