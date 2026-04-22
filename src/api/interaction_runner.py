"""Notebook helpers: list API-tagged routines, describe them, prompt via ``input``, run with ``ScriptTool`` api mode.

Example (run from project root, e.g. ``uv run python``)::

    from src.api.interaction_runner import (
        list_api_interaction_methods ,
        describe_callable_for_ui ,
        prompt_kwargs_with_input ,
        build_decorated_api_shim ,
        demo_trading_available_ports_interactive ,
    )
    from src.api.trading import TradingAPI

    # 1) All routines with a docstring ``[API Interaction]`` block
    list_api_interaction_methods()

    # 2) Description + schema + parameter rows (``override_arg_attr`` merged)
    describe_callable_for_ui(TradingAPI.available_ports)

    # 3-4) Prompt each parameter then execute api-mode ``ScriptTool`` chain
    demo_trading_available_ports_interactive()
"""
from __future__ import annotations

from typing import Any , Callable

import yaml

from src.api.contract import (
    bind_explicit_only ,
    describe_api_callable ,
    iter_endpoints_with_interaction ,
    interaction_for_callable ,
    validate_interaction_schema ,
)
from src.proj.util.script.script_tool import ScriptTool

def list_api_interaction_methods():
    """Return one dict per ``src.api`` routine that defines ``[API Interaction]`` (serializable)."""
    return [rec for rec in iter_endpoints_with_interaction()]

def describe_callable_for_ui(obj : Callable[..., Any]) -> dict[str, Any]:
    """Return ``description``, ``schema``, and ``parameters`` (see :func:`describe_api_callable`)."""
    return describe_api_callable(obj)


def parse_cli_scalar(text : str , * , default : Any) -> Any:
    """Parse a single user line; empty string keeps *default* (YAML for ``null`` / ``true``)."""
    t = text.strip()
    if t == '':
        return default
    return yaml.safe_load(t)


def prompt_kwargs_with_input(obj : Callable[..., Any]) -> dict[str, Any]:
    """Prompt for each explicit (non var-*) parameter; values parsed with :func:`parse_cli_scalar`."""
    desc = describe_api_callable(obj)
    out : dict[str, Any] = {}
    for row in desc['parameters']:
        name = row['name']
        ann_s = row.get('annotation')
        dflt = row['default']
        extra = ''
        if row.get('override'):
            extra = f" override={row['override']!r}"
        raw = input(f"{name} ({ann_s}) default={dflt!r}{extra}: ")
        out[name] = parse_cli_scalar(raw , default = dflt)
    return out


def build_decorated_api_shim(
    underlying : Callable[..., Any] ,
    shim : Callable[..., Any] ,
    task_name : str ,
) -> Callable[..., Any]:
    """
    Wrap *shim* with ``ScriptTool`` in ``source_mode='api'``, using ``[API Interaction]`` from *underlying*.

    *underlying* is only used to read the docstring contract (typically the real ``src.api`` entry);
    *shim* must implement the same explicit parameter names for UI/binding.
    """
    data = interaction_for_callable(underlying)
    if not data:
        raise ValueError(f'underlying {underlying!r} has no [API Interaction] block')
    errs = validate_interaction_schema(data)
    if errs:
        raise ValueError('; '.join(errs))
    lt = data.get('lock_timeout')
    if lt is None:
        lt = 60
    else:
        lt = int(lt)
    tool = ScriptTool(
        task_name ,
        source_mode = 'api' ,
        interaction = data ,
        lock_num = int(data['lock_num']) ,
        lock_timeout = lt ,
        lock_name = task_name ,
    )
    return tool(shim)


def demo_trading_available_ports_interactive() -> list[str]:
    """
    Primary demo: ``TradingAPI.available_ports`` — prompt ``backtest``, run through api-mode ``ScriptTool``.

    Run from project root (``uv run python``) so ``PATH.main`` and task DB paths resolve.
    """
    from src.api.trading import TradingAPI

    def shim(backtest : bool | None = None) -> list[str]:
        return TradingAPI.available_ports(backtest = backtest)

    wrapped = build_decorated_api_shim(TradingAPI.available_ports , shim , 'api_trading_available_ports')
    kwargs = prompt_kwargs_with_input(TradingAPI.available_ports)
    ba = bind_explicit_only(shim , kwargs)
    return wrapped(**ba.arguments)


__all__ = [
    'list_api_interaction_methods' ,
    'describe_callable_for_ui' ,
    'parse_cli_scalar' ,
    'prompt_kwargs_with_input' ,
    'build_decorated_api_shim' ,
    'demo_trading_available_ports_interactive' ,
]
