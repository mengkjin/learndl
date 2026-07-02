"""Shared script-parameter type resolution and value coercion (CLI + Streamlit)."""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias, cast

from src.proj import Logger, Options  # noqa: F401 — used by eval(type_spec)

__all__ = [
    'ScriptParamType',
    'ParamCodec',
    'resolve_param_type',
    'resolve_options',
    'is_options_type_spec',
    'default_value',
    'coerce_value',
    'coerce_cli_input',
    'format_default',
    'remove_extra_prefix',
    'option_to_value',
    'value_to_option',
]

ScriptParamType: TypeAlias = (
    Literal['str', 'int', 'float', 'bool', 'list', 'tuple', 'enum'] | Sequence[str] | str
)


class ParamCodec(Protocol):
    """Minimal parameter descriptor consumed by codec helpers."""

    name: str
    type: ScriptParamType
    default: Any
    prefix: str
    enum: list[str] | None


def remove_extra_prefix(s: str | Any, prefix: str, *, remain_prefix: bool = True) -> str:
    """Strip repeated occurrences of *prefix* from *s*, optionally re-adding it once."""
    text = str(s)
    if not prefix:
        return text
    while text.startswith(prefix):
        text = text.removeprefix(prefix)
    if remain_prefix:
        return f'{prefix}{text}'
    return text


def resolve_param_type(type_spec: ScriptParamType, *, enum: list[str] | None = None) -> type | list[Any]:
    """Resolve YAML ``type`` to a Python type or list of allowed option values."""
    if isinstance(type_spec, str):
        if type_spec == 'str':
            return str
        if type_spec == 'int':
            return int
        if type_spec == 'float':
            return float
        if type_spec == 'bool':
            return bool
        if type_spec in ('list', 'tuple', 'enum'):
            assert enum, f'enum is required for {type_spec}'
            return list(enum)
        try:
            return eval(type_spec)
        except Exception as exc:
            Logger.warning(exc)
            Logger.warning(f'Invalid type: {type_spec} , using str as default')
            return str
    if isinstance(type_spec, (list, tuple)):
        return list(type_spec)
    raise ValueError(f'Invalid type: {type_spec}')


def _options_method_name(type_spec: str) -> str | None:
    """Parse ``Options.available_schedules()`` → ``available_schedules``."""
    stripped = type_spec.strip()
    if not stripped.startswith('Options.'):
        return None
    tail = stripped.removeprefix('Options.').strip()
    if tail.endswith('()'):
        tail = tail[:-2].strip()
    if not tail or '(' in tail:
        return None
    return tail


def _is_options_type_spec(type_spec: ScriptParamType) -> bool:
    return isinstance(type_spec, str) and type_spec.strip().startswith('Options.')


def is_options_type_spec(type_spec: ScriptParamType) -> bool:
    """Return True when YAML ``type`` is an ``Options.available_*()`` provider."""
    return _is_options_type_spec(type_spec)


def resolve_options(param: ParamCodec, *, refresh: bool = True) -> list[Any]:
    """Return selectable options for enum-like parameters."""
    if _is_options_type_spec(param.type):
        assert isinstance(param.type, str)
        dynamic = _call_options_provider(param.type, refresh=refresh)
        if dynamic is not None:
            return dynamic
    ptype = resolve_param_type(param.type, enum=param.enum)
    if isinstance(ptype, list):
        return ptype
    if isinstance(ptype, type) and ptype is bool:
        return [True, False]
    if isinstance(ptype, type) and ptype is str and isinstance(param.type, str):
        dynamic = _call_options_provider(param.type, refresh=refresh)
        if dynamic is not None:
            return dynamic
    return []


def _call_options_provider(type_spec: str, *, refresh: bool) -> list[Any] | None:
    """Evaluate ``Options.available_*()``; when *refresh*, read from :class:`OptionsDefinition`."""
    method_name = _options_method_name(type_spec)
    if method_name is None:
        return None
    if refresh:
        from src.proj.env.options import OptionsDefinition

        definition_provider = getattr(OptionsDefinition, method_name, None)
        if callable(definition_provider):
            try:
                return list(cast(Sequence[Any], definition_provider()))
            except Exception as exc:
                Logger.warning(f'Failed to resolve options from OptionsDefinition.{method_name}: {exc}')
                return None
    try:
        provider = eval(type_spec)
    except Exception:
        return None
    if not callable(provider):
        return None
    try:
        sig = inspect.signature(provider)
        if 'refresh' in sig.parameters:
            result = provider(refresh=refresh)
        else:
            result = provider()
    except Exception as exc:
        Logger.warning(f'Failed to resolve options from {type_spec!r}: {exc}')
        return None
    if result is None:
        return None
    return list(cast(Sequence[Any], result))


def default_value(param: ParamCodec, sig_default: Any = inspect.Parameter.empty) -> Any:
    """Merge YAML ``default`` (with optional prefix) and ``inspect.signature`` default."""
    if param.default is not None and param.default != '':
        return option_to_value(param, param.default)
    if sig_default is not inspect.Parameter.empty:
        return sig_default
    return None


def format_default(param: ParamCodec, sig_default: Any = inspect.Parameter.empty) -> str:
    """Human-readable default for CLI prompts."""
    value = default_value(param, sig_default)
    if value is None:
        return 'None'
    return repr(value)


def option_to_value(param: ParamCodec, option: Any) -> Any:
    """Convert a raw widget/CLI option to a typed Python value."""
    ptype = resolve_param_type(param.type, enum=param.enum)
    if option is None or option == '' or option == 'Choose an option':
        return None
    if isinstance(ptype, list):
        options = ['Choose an option'] + [f'{param.prefix}{entry}' for entry in ptype]
        values: list[Any] = [None, *ptype]
        normalized = option
        if normalized not in options:
            normalized = remove_extra_prefix(normalized, param.prefix, remain_prefix=False)
            for index, entry in enumerate(ptype):
                if str(normalized) == str(entry):
                    return ptype[index]
            raise ValueError(f"Invalid option {option!r} for {param.name}; expected one of {ptype}")
        return values[options.index(option)]
    if ptype is str:
        return option.strip() if option is not None else None
    if ptype is bool:
        if isinstance(option, bool):
            return option
        lowered = str(option).strip().lower()
        if lowered in ('true', 'yes', 'y', '1'):
            return True
        if lowered in ('false', 'no', 'n', '0'):
            return False
        raise ValueError(f"Invalid bool value {option!r} for {param.name}")
    if ptype is int:
        return int(option)
    if ptype is float:
        return float(option)
    raise ValueError(f'Unsupported param type for {param.name}: {ptype}')


def value_to_option(param: ParamCodec, value: Any) -> Any:
    """Convert a typed Python value to a widget/CLI option string."""
    ptype = resolve_param_type(param.type, enum=param.enum)
    if value is None or value == '' or value == 'Choose an option':
        return 'Choose an option' if isinstance(ptype, list) else None
    if isinstance(ptype, list):
        options = ['Choose an option'] + [f'{param.prefix}{entry}' for entry in ptype]
        values = [None, *[str(entry) for entry in ptype]]
        normalized = value
        if normalized not in values and normalized not in ptype:
            normalized = remove_extra_prefix(normalized, param.prefix, remain_prefix=False)
        lookup = str(normalized)
        if lookup not in values:
            for index, entry in enumerate(ptype):
                if str(entry) == lookup or entry == normalized:
                    return options[index + 1]
            raise ValueError(f"Invalid value {value!r} for {param.name}")
        return options[values.index(lookup)]
    if ptype is str:
        return None if value is None or value == '' else str(value).strip()
    if ptype is bool:
        return value if isinstance(value, bool) else bool(value)
    if ptype is int:
        return None if value is None else int(value)
    if ptype is float:
        return None if value is None else float(value)
    raise ValueError(f'Unsupported param type for {param.name}: {ptype}')


def coerce_cli_input(
    param: ParamCodec,
    raw: str,
    *,
    sig_default: Any = inspect.Parameter.empty,
    force_input: bool = False,
) -> Any:
    """Coerce CLI text input; empty means default unless *force_input*."""
    stripped = raw.strip()
    if not stripped:
        if force_input:
            raise ValueError('A value is required')
        return default_value(param, sig_default)
    lowered = stripped.lower()
    if lowered in ('null', 'none'):
        return None
    if stripped in ('""', "''"):
        return ''
    return coerce_value(param, stripped, sig_default=sig_default)


def coerce_value(
    param: ParamCodec,
    raw: Any,
    *,
    sig_default: Any = inspect.Parameter.empty,
) -> Any:
    """Coerce user input; empty values fall back to :func:`default_value`."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default_value(param, sig_default)
    ptype = resolve_param_type(param.type, enum=param.enum)
    if isinstance(ptype, list) or ptype is bool:
        return option_to_value(param, raw)
    if ptype is str:
        return str(raw).strip()
    if ptype is int:
        return int(raw)
    if ptype is float:
        return float(raw)
    raise ValueError(f'Unsupported param type for {param.name}: {ptype}')
