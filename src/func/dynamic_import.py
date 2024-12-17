import inspect
import importlib.util

from collections.abc import Iterable
from pathlib import Path
from typing import Callable


def dynamic_modules(path : str | Path):
    if isinstance(path , str): path = Path(path)
    paths = path.rglob('*.py') if path.is_dir() else [path]
    for p in paths:
        spec = importlib.util.spec_from_file_location(str(p) , p)
        assert spec is not None and spec.loader is not None , f'{p} is not a valid module'
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module

def dynamic_members(path : str | Path , predicate : Callable):
    for module in dynamic_modules(path):
        for name , obj in inspect.getmembers(module , predicate):
            yield name , obj

def true_subclass(cls , base_cls):
    return (
        inspect.isclass(cls) and 
        issubclass(cls , base_cls) and 
        (cls is not base_cls) and 
        (not getattr(cls , '__abstractmethods__' , None))
    )
        