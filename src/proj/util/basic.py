"""Basic utilities in Project level: Iterable filtering, temp file context, and dot-key nested dict flattening."""
from __future__ import annotations

from typing import Any, Callable , Iterable
from pathlib import Path
from pprint import pformat
import os

from src.proj.log import Logger
from src.proj.env import PATH

__all__ = ['FilteredIterable' , 'TempFile' , 'FlattenDict']

class FilteredIterable:
    """Iterator that yields only items passing a callable or parallel boolean stream."""

    def __init__(self, iterable, condition : Callable | Iterable | None = None , **kwargs):
        """If ``condition`` is iterable, zip with items; if callable, filter by predicate."""
        self.iterable  = iter(iterable)
        if condition is None:
            self.condition = lambda x: True
        elif callable(condition):
            self.condition = condition
        else:
            self.condition = iter(condition)
        self.kwargs = kwargs
    def __iter__(self):
        """Return self as iterator."""
        return self
    def __next__(self):
        """Skip items until ``condition`` is truthy."""
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: 
                return item

class TempFile:
    """Context manager that deletes a filesystem path on exit."""

    def __init__(self, file_name: str):
        """Store path to remove in ``__exit__``."""
        self.file_name = file_name

    def __enter__(self):
        """Return the temp path string."""
        return self.file_name

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Best-effort ``os.remove`` with error logging."""
        try:
            os.remove(self.file_name)
        except Exception as e:
            Logger.error(f'Failed to remove temp file: {e}')

class FlattenDict:
    """Dot-key flattened view over nested dicts with optional selective nesting."""

    def __init__(self, input: FlattenDict | dict[str, Any] | None = None , * , keep_nested : Callable[[str], bool] | None = None , **kwargs):
        """Build ``flattened`` from dict or another ``FlattenDict``; ``keep_nested`` preserves subtrees."""
        self.raw = input.raw if isinstance(input, FlattenDict) else input or {}
        self.keep_nested = keep_nested
        self.flattened : dict[str, Any] = self.flatten_dict(self.raw , keep_nested = self.keep_nested)

    def __bool__(self):
        """True if flattened map is non-empty."""
        return bool(self.flattened)

    def items(self):
        """Key/value pairs of the flattened map."""
        return self.flattened.items()

    def keys(self):
        """Dot keys of the flattened map."""
        return self.flattened.keys()

    def values(self):
        """Values of the flattened map."""
        return self.flattened.values()

    def __repr__(self) -> str:
        """Pretty-printed flattened dict."""
        return pformat(self.flattened)

    def __str__(self) -> str:
        """String form of flattened dict."""
        return str(self.flattened)

    def __len__(self):
        """Number of flattened keys."""
        return len(self.flattened)

    def __getitem__(self, key: str):
        """Scalar value or nested dict reconstructed for ``key.*`` subtree."""
        if key in self.flattened:
            return self.flattened[key]
        else:
            out = self.find_sub_dict(self.flattened, key)
            assert out , f'{key} not found in {self.keys()}'
            return out

    def __setitem__(self, key: str, value: Any):
        """Set an existing top-level flattened key (no implicit insert)."""
        assert key in self.flattened , f'{key} not found in {self.keys()} , cannot set value , use update with relevant_only = False instead'
        self.flattened[key] = value

    def update(self, d : 'dict[str, Any] | FlattenDict' , relevant_only: bool = True):
        """Merge flattened keys from ``d``; unknown keys skipped when ``relevant_only``."""
        flattened_input = self.flatten_dict(d , keep_nested = self.keep_nested)
        for k, v in flattened_input.items():
            if k in self.flattened or not relevant_only:
                self.flattened[k] = v  
        return self

    def get(self, key: str, default: Any = None):
        """Get leaf or subtree like ``__getitem__``, else ``default``."""
        if key in self.flattened:
            return self.flattened[key]
        else:
            out = self.find_sub_dict(self.flattened, key)
            return out or default

    def __contains__(self, key: str):
        """True if ``key`` is a prefix of any flattened path."""
        return key in self.flattened or any(k.startswith(f'{key}.') for k in self.keys())

    @classmethod
    def find_sub_dict(cls, d: dict[str, Any] , key: str):
        """Nested dict from keys ``key.*`` under flattened ``d``."""
        return cls.nested_dict({k.removeprefix(f'{key}.'): v for k, v in d.items() if k.startswith(f'{key}.')})

    @classmethod
    def flatten_dict(cls , d: 'dict | FlattenDict' , prefix: str = '' , * , keep_nested: Callable[[str], bool] | None = None):
        """Recursively flatten dicts to dot keys unless ``keep_nested`` keeps a prefix."""
        if isinstance(d, FlattenDict):
            d = d.raw
        target = {}
        for k, v in d.items():
            new_k = f'{prefix}{k}'
            if isinstance(v, dict) and v and not (keep_nested and keep_nested(new_k)):
                target.update(cls.flatten_dict(v, f'{new_k}.' , keep_nested = keep_nested))
            else:
                target[new_k] = v
        return target

    @classmethod
    def nested_dict(cls, flattened: dict[str, Any]):
        """Invert ``flatten_dict``: dot keys become nested dicts."""
        target = {}
        for k, v in flattened.items():
            entries = k.split('.')
            obj = target
            for i in range(len(entries) - 1):
                if entries[i] not in obj:
                    obj[entries[i]] = {}
                obj = obj[entries[i]]
            obj[entries[-1]] = v
        return target

    def to_dict(self):
        """Return the flattened mapping (alias of internal store)."""
        return self.flattened

    def dump_yaml(self, path: Path , overwrite: bool = False , vb_level: Any = 1):
        """Write flattened dict to YAML via ``PATH.dump_yaml``."""
        if path.exists() and not overwrite:
            Logger.alert1(f'{path} already exists' , vb_level = vb_level)
            return
        if overwrite:
            path.unlink(missing_ok=True)
        PATH.dump_yaml(self.flattened, path)

    @classmethod
    def from_input(cls, input: 'dict | Path | list[Path] | FlattenDict | None' , * , keep_nested: Callable[[str], bool] | None = None):
        """Factory: dict/FlattenDict, single YAML path, or list of YAMLs keyed by stem."""
        if input is None or isinstance(input, (FlattenDict, dict)):
            return cls(input , keep_nested = keep_nested)
        elif isinstance(input, Path):
            return cls.from_yaml(input , keep_nested = keep_nested)
        elif isinstance(input, list):
            return cls.from_yamls(input , keep_nested = keep_nested)
        else:
            raise ValueError(f"Invalid input type: {type(input)}")

    @classmethod
    def from_yaml(cls, path: Path , * , keep_nested: Callable[[str], bool] | None = None):
        """Load one YAML file into a ``FlattenDict``."""
        return cls(PATH.read_yaml(path) , keep_nested = keep_nested)

    @classmethod
    def from_yamls(cls, paths: list[Path] , * , keep_nested: Callable[[str], bool] | None = None):
        """Load multiple YAMLs into nested dict keyed by each file's stem, then flatten."""
        input_dict = {}
        for path in paths:
            input_dict[path.stem] = PATH.read_yaml(path)
        return cls(input_dict , keep_nested = keep_nested)