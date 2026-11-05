"""Script parameter schema: YAML header merged with ``main`` signature defaults."""
from __future__ import annotations

import ast
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
    def from_source(cls, path: Path) -> ScriptParamSchema | None:
        """Read literal defaults without executing a script; None requests runtime fallback.

        Unknown decorators and computed defaults retain the callable-based path so
        their signature semantics are never guessed during parameter collection.
        """
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
        if len(mains) != 1:
            return None
        main = mains[0]
        for decorator in main.decorator_list:
            if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == 'ScriptTool'):
                return None
        positional = [*main.args.posonlyargs, *main.args.args]
        pairs = list(zip(positional[len(positional) - len(main.args.defaults):], main.args.defaults))
        pairs.extend((arg, value) for arg, value in zip(main.args.kwonlyargs, main.args.kw_defaults)
                     if value is not None)
        try:
            defaults = {arg.arg: ast.literal_eval(value) for arg, value in pairs}
        except (ValueError, TypeError):
            return None
        header = ScriptHeader.read_from_file(path)
        return cls(path, header.get_param_inputs(), defaults)

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
