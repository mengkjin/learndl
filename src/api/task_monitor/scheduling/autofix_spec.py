"""Side-effect-free autofix parsing. Never import submitted scripts or eval types."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import yaml

ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$')
RESERVED = {'source', 'task_id', 'email_recipient', 'forfeit_if_done', 'debug_mode', 'max_vb'}


class ManifestError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key)
        if not isinstance(key, str) or key in result:
            raise ManifestError(f'Duplicate or non-string YAML key: {key!r}')
        result[key] = loader.construct_object(value)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def manifest(path: Path) -> list[dict]:
    try:
        doc = yaml.load(path.read_text(), Loader=UniqueLoader)
        if not isinstance(doc, dict) or set(doc) != {'schema_version', 'fixes'}:
            raise ManifestError('Expected schema_version and fixes only')
        if type(doc['schema_version']) is not int or doc['schema_version'] != 1 or not isinstance(doc['fixes'], list):
            raise ManifestError('Expected schema_version: 1 and fixes: []')
        seen = set()
        for item in doc['fixes']:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not ID.fullmatch(item['id']):
                raise ManifestError('Every fix needs a valid unique id')
            if item['id'] in seen:
                raise ManifestError(f'Duplicate id: {item["id"]}')
            seen.add(item['id'])
            # Unsupported values must not reach JSON persistence or hashing.
            canonical(item)
        return doc['fixes']
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ManifestError(str(exc)) from exc


def declaration(item: dict) -> dict:
    """Freeze unknown fields too; correcting an invalid declaration needs a new ID."""
    result = {k: v for k, v in item.items() if k not in {'id', 'description', 'enabled'}}
    params = result.get('params', {})
    if isinstance(params, dict) and type(params.get('email', True)) is bool:
        result['params'] = dict(params, email=True)
    return result


def timeout_seconds(value='12h') -> float:
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(s|m|h|d)', str(value))
    if not match or float(match[1]) <= 0:
        raise ValueError('timeout must be a positive duration, e.g. 12h')
    seconds = float(match[1]) * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[match[2]]
    if not math.isfinite(seconds):
        raise ValueError('timeout must be finite')
    return seconds


def script_path(root: Path, name: str | None) -> Path:
    if not isinstance(name, str):
        raise ValueError('script must be a relative Python path')
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'scripts' or relative.suffix != '.py':
        raise ValueError('script must be a relative scripts/**/*.py path without ..')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Script paths may not contain symbolic links')
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve() / 'scripts'):
        raise FileNotFoundError(f'Script does not exist inside scripts: {name}')
    return current


def header(text: str) -> dict:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(('#!', '# coding:')):
            continue
        if not line.startswith('#'):
            break
        lines.append(line[1:])
    value = yaml.load('\n'.join(lines), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError('Script requires YAML header metadata')
    return value


def _typed(name, value, spec):
    kind = spec.get('type', 'str')
    if isinstance(kind, list):
        if not any(type(value) is type(option) and value == option for option in kind):
            raise ValueError(f'{name}: not one of the declared choices')
    elif kind in ('str', 'int', 'bool', 'float'):
        expected = {'str': str, 'int': int, 'bool': bool, 'float': float}[kind]
        if type(value) is not expected and not (kind == 'float' and type(value) is int):
            raise ValueError(f'{name}: expected {kind}')
    else:
        raise ValueError(f'{name}: autofix requires a static scalar type or literal choices')
    if not isinstance(value, (str, int, float, bool)) or isinstance(value, str) and ('\0' in value or '\n' in value):
        raise ValueError(f'{name}: unsupported parameter value')


def validate(root: Path, item: dict) -> dict:
    extra = set(item) - {'id', 'script', 'params', 'description', 'enabled', 'timeout', 'blocking'}
    if extra:
        raise ValueError(f'Unknown task fields: {sorted(extra)}; use a new ID for every retry')
    if type(item.get('enabled', True)) is not bool or not isinstance(item.get('description', ''), str):
        raise ValueError('enabled must be boolean; description must be text')
    blocking = item.get('blocking', 'none')
    if blocking not in {'none', 'research', 'research_and_data'}:
        raise ValueError('Invalid autofix blocking type')
    path = script_path(root, item.get('script'))
    text = path.read_text()
    metadata = header(text)
    if metadata.get('autofix') is not True or metadata.get('disabled', False):
        raise ValueError('Script must explicitly declare autofix: true and not be disabled')
    tree = ast.parse(text, filename=str(path))
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    if main is None or not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'ScriptTool' for n in main.decorator_list):
        raise ValueError('Autofix requires a main function decorated with ScriptTool')
    # runpy executes the module as __main__; a decorator alone never runs a task.
    # Reject this omission before spawning a process which would exit 0 without
    # producing ScriptTool's result/notification handoff.
    def is_main_guard(node):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            return False
        test = node.test
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
            return False
        left, right = test.left, test.comparators[0]
        return ((isinstance(left, ast.Name) and left.id == '__name__' and
                 isinstance(right, ast.Constant) and right.value == '__main__') or
                (isinstance(right, ast.Name) and right.id == '__name__' and
                 isinstance(left, ast.Constant) and left.value == '__main__'))

    if not any(is_main_guard(node) and any(
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and
            isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'main'
            for stmt in node.body) for node in tree.body if isinstance(node, ast.If)):
        raise ValueError("Autofix requires an executable if __name__ == '__main__': main() entrypoint")
    schema = metadata.get('parameters', {})
    params = item.get('params', {})
    if not isinstance(schema, dict) or not isinstance(params, dict):
        raise ValueError('parameters and params must be mappings')
    if set(params) & RESERVED or set(params) - set(schema) - {'email'}:
        raise ValueError('Unknown or reserved parameter names: ' + ', '.join(sorted(set(params) & RESERVED | (set(params) - set(schema) - {'email'}))))
    # Only literal defaults are permitted; no evaluation or importing the script.
    names = [n.arg for n in main.args.args]
    defaults = dict(zip(names[len(names) - len(main.args.defaults):], main.args.defaults))
    defaults.update({n.arg: v for n, v in zip(main.args.kwonlyargs, main.args.kw_defaults) if v is not None})
    effective = dict(params)
    for name, spec in schema.items():
        if (not isinstance(spec, dict) or name in RESERVED or name == 'email'
                or not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', name)
                or name not in names + [n.arg for n in main.args.kwonlyargs]):
            raise ValueError(f'Invalid/reserved parameter declaration: {name}')
        if name not in effective:
            if spec.get('required', False):
                raise ValueError(f'Missing required parameter: {name}')
            if 'default' in spec:
                effective[name] = spec['default']
            elif name in defaults:
                effective[name] = ast.literal_eval(defaults[name])
            else:
                raise ValueError(f'{name}: declare an explicit static default')
        _typed(name, effective[name], spec)
    required = names[:len(names) - len(main.args.defaults)] + [n.arg for n, d in zip(main.args.kwonlyargs, main.args.kw_defaults) if d is None]
    if set(required) - set(effective):
        raise ValueError('Required main parameters missing from static schema')
    effective.setdefault('email', True)
    if type(effective['email']) is not bool:
        raise ValueError('email must be a YAML boolean')
    effective['email'] = True  # Autofix always enables the script's result mail.
    args = [f'--{k}={v!r}' for k, v in effective.items()]
    return {'script': str(path.relative_to(root)), 'params': effective, 'args': args,
            'blocking': blocking,
            'timeout_seconds': min(timeout_seconds(item.get('timeout', '12h')), 3600) if blocking != 'none' else timeout_seconds(item.get('timeout', '12h')),
            'script_sha256': hashlib.sha256(text.encode()).hexdigest()}


def redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if any(s in k.lower() for s in ('password', 'token', 'secret', 'credential', 'api_key')) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
