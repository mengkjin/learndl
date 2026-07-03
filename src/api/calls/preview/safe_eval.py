"""Restricted expression evaluator for previewing loaded objects."""

from __future__ import annotations

import ast
import builtins
from typing import Any

from src.proj import Logger

__all__ = ['SafeObjEval']

_MAX_EXPR_LEN = 200

_ALLOWED_NODE_TYPES: frozenset[type[ast.AST]] = frozenset({
    ast.Expression,
    ast.Attribute,
    ast.Call,
    ast.Name,
    ast.Subscript,
    ast.Constant,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Load,
    ast.keyword,
})

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({
    'len',
    'type',
    'repr',
    'str',
    'int',
    'float',
    'bool',
    'list',
    'tuple',
    'dict',
    'set',
    'frozenset',
    'bytes',
    'min',
    'max',
    'sum',
    'abs',
    'round',
    'sorted',
    'any',
    'all',
    'callable',
    'hash',
    'format',
    'isinstance',
    'issubclass',
    'iter',
    'next',
    'enumerate',
    'zip',
    'reversed',
    'slice',
    'chr',
    'ord',
    'divmod',
    'pow',
    'id',
    'hex',
    'oct',
    'bin',
    'range',
    'map',
    'filter',
})

_SAFE_TYPE_NAMES: frozenset[str] = frozenset({
    'object',
})

_ALLOWED_NAMES: frozenset[str] = frozenset({'obj'}) | _SAFE_BUILTIN_NAMES | _SAFE_TYPE_NAMES

_BLOCKED_METHOD_NAMES: frozenset[str] = frozenset({
    'save',
    'write',
    'to_csv',
    'to_feather',
    'to_parquet',
    'to_pickle',
    'to_json',
    'to_excel',
    'to_hdf',
    'to_sql',
    'to_numpy',
    'dump',
    'unlink',
    'rmdir',
    'chmod',
    'chown',
    'exec',
    'eval',
    'open',
    'system',
    'remove',
    'rename',
    'replace',
    'mkdir',
    'touch',
    'rmtree',
    'copy',
    'move',
    'send',
    'put',
    'delete',
    'drop',
    'update',
    'insert',
    'commit',
    'execute',
    'run',
    'spawn',
    'popen',
    'compile',
    'globals',
    'locals',
    'vars',
    'dir',
    'getattr',
    'setattr',
    'delattr',
    'input',
    '__import__',
})


class SafeObjEval:
    """Evaluate read-only ``obj.*`` and whitelisted builtin expressions."""

    @classmethod
    def eval(cls, expr: str, obj: Any) -> Any:
        """Parse, validate, and evaluate *expr* against *obj*."""
        cleaned = expr.strip()
        if not cleaned:
            raise ValueError('Expression must not be empty')
        if len(cleaned) > _MAX_EXPR_LEN:
            raise ValueError(f'Expression too long (max {_MAX_EXPR_LEN} characters)')

        tree = ast.parse(cleaned, mode='eval')
        cls._validate(tree)
        result = eval(
            compile(tree, '<preview>', 'eval'),
            {'__builtins__': {}},
            cls._evaluation_globals(obj),
        )
        Logger.display(result)
        return result

    @classmethod
    def _evaluation_globals(cls, obj: Any) -> dict[str, Any]:
        namespace: dict[str, Any] = {'obj': obj}
        for name in _ALLOWED_NAMES:
            if name == 'obj':
                continue
            namespace[name] = getattr(builtins, name)
        return namespace

    @classmethod
    def _validate(cls, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_NODE_TYPES:
                raise ValueError(f'Disallowed syntax: {type(node).__name__}')
            if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
                raise ValueError(f'Disallowed name: {node.id!r}')
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    raise ValueError(f'Disallowed attribute: {node.attr!r}')
            if isinstance(node, ast.Call):
                cls._validate_call(node)

    @classmethod
    def _validate_call(cls, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            method_name = func.attr
            if method_name in _BLOCKED_METHOD_NAMES:
                raise ValueError(f'Disallowed method: {method_name!r}')
            if method_name.startswith('_'):
                raise ValueError(f'Disallowed method: {method_name!r}')
            return
        if isinstance(func, ast.Name):
            if func.id not in _SAFE_BUILTIN_NAMES:
                raise ValueError(f'Disallowed function: {func.id!r}')
            return
        raise ValueError(
            'Only attribute calls and safe builtins are allowed (e.g. obj.head(), len(obj))',
        )
