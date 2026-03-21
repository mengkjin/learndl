from typing import Any, Callable , Iterable , Literal
from pathlib import Path
from pprint import pformat
import os

from src.proj.log import Logger
from src.proj.env import PATH

__all__ = ['FilteredIterable' , 'TempFile' , 'FlattenDict']

class FilteredIterable:
    def __init__(self, iterable, condition : Callable | Iterable | None = None , **kwargs):
        self.iterable  = iter(iterable)
        if condition is None:
            self.condition = lambda x: True
        elif callable(condition):
            self.condition = condition
        else:
            self.condition = iter(condition)
        self.kwargs = kwargs
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: 
                return item

class TempFile:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def __enter__(self):
        return self.file_name

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            os.remove(self.file_name)
        except Exception as e:
            Logger.error(f'Failed to remove temp file: {e}')

class FlattenDict:
    def __init__(self, input: 'FlattenDict | dict[str, Any] | None' = None , * , keep_nested : Callable[[str], bool] | None = None , **kwargs):
        self.raw = input.raw if isinstance(input, FlattenDict) else input or {}
        self.keep_nested = keep_nested
        self.flattened : dict[str, Any] = self.flatten_dict(self.raw , keep_nested = self.keep_nested)

    def __bool__(self):
        return bool(self.flattened)

    def items(self):
        return self.flattened.items()

    def keys(self):
        return self.flattened.keys()

    def values(self):
        return self.flattened.values()

    def __repr__(self) -> str:
        return pformat(self.flattened)

    def __str__(self) -> str:
        return str(self.flattened)

    def __len__(self):
        return len(self.flattened)

    def __getitem__(self, key: str):
        if key in self.flattened:
            return self.flattened[key]
        else:
            out = self.find_sub_dict(self.flattened, key)
            assert out , f'{key} not found in {self.keys()}'
            return out

    def __setitem__(self, key: str, value: Any):
        assert key in self.flattened , f'{key} not found in {self.keys()} , cannot set value , use update with relevant_only = False instead'
        self.flattened[key] = value

    def update(self, d : 'dict[str, Any] | FlattenDict' , relevant_only: bool = True):
        flattened_input = self.flatten_dict(d , keep_nested = self.keep_nested)
        for k, v in flattened_input.items():
            if k in self.flattened or not relevant_only:
                self.flattened[k] = v  
        return self

    def get(self, key: str, default: Any = None):
        if key in self.flattened:
            return self.flattened[key]
        else:
            out = self.find_sub_dict(self.flattened, key)
            return out or default

    def __contains__(self, key: str):
        return key in self.flattened or any(k.startswith(f'{key}.') for k in self.keys())

    @classmethod
    def find_sub_dict(cls, d: dict[str, Any] , key: str):
        return cls.nested_dict({k.removeprefix(f'{key}.'): v for k, v in d.items() if k.startswith(f'{key}.')})

    @classmethod
    def flatten_dict(cls , d: 'dict | FlattenDict' , prefix: str = '' , * , keep_nested: Callable[[str], bool] | None = None):
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
        return self.flattened

    def dump_yaml(self, path: Path , overwrite: bool = False , vb_level: int | Literal['max','min','inf'] = 1):
        if path.exists() and not overwrite:
            Logger.alert1(f'{path} already exists' , vb_level = vb_level)
            return
        if overwrite:
            path.unlink(missing_ok=True)
        PATH.dump_yaml(self.flattened, path)

    @classmethod
    def from_input(cls, input: 'dict | Path | list[Path] | FlattenDict | None' , * , keep_nested: Callable[[str], bool] | None = None):
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
        return cls(PATH.read_yaml(path) , keep_nested = keep_nested)

    @classmethod
    def from_yamls(cls, paths: list[Path] , * , keep_nested: Callable[[str], bool] | None = None):
        input_dict = {}
        for path in paths:
            input_dict[path.stem] = PATH.read_yaml(path)
        return cls(input_dict , keep_nested = keep_nested)