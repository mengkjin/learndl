import os
import importlib.util
import inspect

from src.factor.perf.api import PerfManager
from src.factor.fmp.api import FmpManager
from src.basic import PATH

def perf_test():
    pm = PerfManager.random_test(nfactor=1)
    return pm

def fmp_test():
    pm = FmpManager.random_test(nfactor=1 , verbosity=2)
    return pm

def factor_hierarchy():
    folder_path =  PATH.main.joinpath('src_factors')
    classes = []
    for level_path in folder_path.iterdir():
        if not level_path.is_dir(): continue
        for file_path in level_path.iterdir():
            if file_path.suffix != '.py': continue
            spec_name = f'{level_path.stem}.{file_path.stem}'
            
            spec = importlib.util.spec_from_file_location(spec_name, file_path)
            assert spec is not None and spec.loader is not None , f'{file_path} is not a valid module'
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == spec_name:
                    classes.append((level_path.stem , name ,obj))
    return classes

