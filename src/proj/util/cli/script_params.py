"""CLI prompts for script parameters defined in ScriptHeader YAML."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.proj.log import Logger
from src.proj.util.cli.ask import AskFlag, AskFor
from src.proj.util.cli.prompts import prompt_confirm
from src.proj.util.script.param_codec import (
    coerce_cli_input,
    default_value,
    format_default,
    is_options_type_spec,
    resolve_options,
    resolve_param_type,
)
from src.proj.util.script.param_schema import ScriptParamSchema

__all__ = ['prompt_script_kwargs', 'run_script_interactive']


def _print_param_summary(
    schema: ScriptParamSchema,
    *,
    preset: dict[str, Any],
) -> None:
    Logger.stdout('Script parameters:', color='lightcyan')
    for param in schema.params:
        if param.name in preset:
            Logger.stdout(
                f'{param.name}: {preset[param.name]!r} (fixed)',
                indent=1,
            )
            continue
        sig_default = schema.sig_default(param.name)
        required_note = ' (required)' if param.required else ''
        Logger.stdout(
            f'{param.name}{required_note}: {param.desc or param.name} — default {format_default(param, sig_default)}',
            indent=1,
        )


def _is_missing_required_default(param: Any, schema: ScriptParamSchema) -> bool:
    if not param.required:
        return False
    return default_value(param, schema.sig_default(param.name)) in (None, '')


def _build_default_kwargs(
    schema: ScriptParamSchema,
    *,
    preset: dict[str, Any],
) -> dict[str, Any]:
    kwargs = dict(preset)
    for param in schema.params:
        if param.name in kwargs:
            continue
        value = default_value(param, schema.sig_default(param.name))
        if param.required and value in (None, ''):
            raise ValueError(f'Required parameter [{param.name}] has no default')
        kwargs[param.name] = value
    return kwargs


def _prompt_param_value(
    schema: ScriptParamSchema,
    param_name: str,
    *,
    force_input: bool = False,
) -> AskFlag[Any]:
    param = next(p for p in schema.params if p.name == param_name)
    sig_default = schema.sig_default(param.name)
    ptype = resolve_param_type(param.type, enum=param.enum)
    title = param.desc or param.name.replace('_', ' ').title()
    default_display = format_default(param, sig_default)

    if isinstance(ptype, list) or ptype is bool:
        options = resolve_options(param, refresh=True)
        if not options:
            options = [True, False] if ptype is bool else list(ptype)
        flag = AskFor.Options(
            options,
            confirm=False,
            multiple=False,
            use_checkbox=True if is_options_type_spec(param.type) else None,
            title=title,
            print_options=False,
            help_description=param.desc or f'Value for script parameter [{param.name}].',
        )
        if not flag.valid:
            return flag
        if flag.result is None:
            return AskFlag('invalid')
        return AskFlag('valid').set_result([flag.result])

    empty_hint = 'empty for default' if not force_input else 'required'
    extra_hint = (
        ''
        if force_input
        else '; null/none → None; "" or \'\' → empty string'
    )
    while True:
        flag = AskFor.String(
            title=f'{title} (default: {default_display}; {empty_hint}{extra_hint})',
            help_description=param.desc or f'Enter a value for [{param.name}].',
        )
        if not flag.valid:
            return flag
        assert flag.result is not None
        raw = flag.result
        try:
            value = coerce_cli_input(
                param,
                raw,
                sig_default=sig_default,
                force_input=force_input,
            )
        except (TypeError, ValueError) as exc:
            Logger.error(f'Invalid value for [{param.name}]: {exc}')
            continue
        if force_input and value in (None, ''):
            Logger.error(f'[{param.name}] is required')
            continue
        return AskFlag('valid').set_result([value])


def _prompt_required_params(
    schema: ScriptParamSchema,
    *,
    preset: dict[str, Any],
    skip: frozenset[str],
) -> AskFlag[dict[str, Any]]:
    kwargs = dict(preset)
    for param in schema.params:
        if param.name in kwargs or param.name in skip:
            continue
        if not _is_missing_required_default(param, schema):
            continue
        flag = _prompt_param_value(schema, param.name, force_input=True)
        if not flag.valid:
            return flag
        kwargs[param.name] = flag.result
    return AskFlag('valid').set_result([kwargs])


def _prompt_sequential_kwargs(
    schema: ScriptParamSchema,
    *,
    preset: dict[str, Any],
    skip: frozenset[str],
) -> AskFlag[dict[str, Any]]:
    kwargs = dict(preset)
    for param in schema.params:
        if param.name in kwargs or param.name in skip:
            continue
        force_input = _is_missing_required_default(param, schema)
        flag = _prompt_param_value(schema, param.name, force_input=force_input)
        if not flag.valid:
            return flag
        kwargs[param.name] = flag.result

    for param in schema.params:
        if param.name in kwargs:
            continue
        if param.required:
            Logger.error(f'Required parameter [{param.name}] was not provided')
            return AskFlag('invalid')
        kwargs[param.name] = default_value(param, schema.sig_default(param.name))

    return AskFlag('valid').set_result([kwargs])


def prompt_script_kwargs(
    schema: ScriptParamSchema,
    *,
    preset: dict[str, Any] | None = None,
    skip: frozenset[str] = frozenset(),
    help_description: str = '',
    extra_help_lines: tuple[str, ...] = (),
) -> AskFlag[dict[str, Any]]:
    """Collect kwargs for a script via default-or-customize CLI flow."""
    if not AskFor.check_interactive():
        return AskFlag('exit')

    preset_values = dict(preset or {})
    param_names = [param.name for param in schema.params if param.name not in preset_values]
    param_help = {
        param.name: param.desc
        for param in schema.params
        if param.name not in preset_values and param.desc
    }
    with AskFor._help_scope(
        title='Script parameters',
        help_description=help_description or 'Configure how the script should run.',
        options=param_names,
        option_help=param_help,
        extra_lines=extra_help_lines,
    ):
        _print_param_summary(schema, preset=preset_values)

        required_flag = _prompt_required_params(schema, preset=preset_values, skip=skip)
        if not required_flag.valid or required_flag.result is None:
            return required_flag
        preset_values = required_flag.result

        remaining = [
            param.name
            for param in schema.params
            if param.name not in preset_values and param.name not in skip
        ]
        if not remaining:
            return AskFlag('valid').set_result([preset_values])

        use_defaults = prompt_confirm('Use default parameters for the remaining fields?', default=True)
        if use_defaults is None:
            return AskFlag('exit')
        if use_defaults:
            try:
                kwargs = _build_default_kwargs(schema, preset=preset_values)
            except ValueError as exc:
                Logger.error(str(exc))
                return AskFlag('invalid')
            return AskFlag('valid').set_result([kwargs])

        return _prompt_sequential_kwargs(schema, preset=preset_values, skip=skip)


def run_script_interactive(
    script_key: str,
    *,
    preset: dict[str, Any] | None = None,
    main: Callable[..., Any] | None = None,
) -> AskFlag[dict[str, Any]]:
    """Load a pipeline script, prompt for kwargs, execute when valid."""
    from src.api.util.backend import ScriptRunner
    from src.proj.util.filesys.dynamic_import import dynamic_modules

    runner = ScriptRunner.from_key(script_key)
    resolved_main = main
    if resolved_main is None:
        for module in dynamic_modules(runner.script):
            resolved_main = module.main
            break
        if resolved_main is None:
            raise FileNotFoundError(f'Script main not found: {runner.script}')

    schema = ScriptParamSchema.from_script(Path(runner.script), main=resolved_main)
    flag = prompt_script_kwargs(schema, preset=preset)
    if flag.valid and flag.result is not None:
        resolved_main(**flag.result)
    return flag
