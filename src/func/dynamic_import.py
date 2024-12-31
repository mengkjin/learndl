import inspect
import importlib.util

from pathlib import Path
from typing import Callable , Type


def dynamic_modules(path : str | Path):
    if isinstance(path , str): path = Path(path)
    paths = path.rglob('*.py') if path.is_dir() else [path]
    for p in sorted(paths):
        spec = importlib.util.spec_from_file_location(str(p) , p)
        assert spec is not None and spec.loader is not None , f'{p} is not a valid module'
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module

def dynamic_members(path : str | Path , 
                    predicate : Callable | None = lambda x: inspect.isclass(x) or inspect.isfunction(x) ,
                    subclass_of : Type | None = None ,
                    ignore_imported_members = True
                    ):
    for module in dynamic_modules(path):
        for name , obj in inspect.getmembers(module , predicate):
            if ignore_imported_members and module.__name__ !=  getattr(obj , '__module__' , None): continue
            if subclass_of is not None and not true_subclass(obj , subclass_of): continue
            yield name , obj

def true_subclass(cls , base_cls):
    return (
        inspect.isclass(cls) and 
        issubclass(cls , base_cls) and 
        (cls is not base_cls) and 
        (not getattr(cls , '__abstractmethods__' , None))
    )