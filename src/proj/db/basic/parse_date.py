"""Extract date from path and directory."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.proj.core import strPath , strPaths
from src.proj.cal import Dates

__all__ = ['path_date' , 'file_dates' , 'dir_dates' , 'load_df_max_date' , 'load_df_min_date']

def path_date(path : strPath) -> int:
    """get date from path"""
    return int(Path(path).stem[-8:])

def file_dates(paths : strPaths , startswith = '' , endswith = '') -> Dates:
    """get dates from paths"""
    if isinstance(paths , Mapping):
        paths = paths.values()
    paths = [Path(p) for p in paths]
    datestrs = [p.stem[-8:] for p in paths if p.name.startswith(startswith) and p.name.endswith(endswith)]
    return Dates([int(ds) for ds in datestrs if ds.isdigit() and len(ds) == 8])

def dir_dates(directory : Path , start = None , end = None , year = None) -> Dates:
    """get dates from directory"""
    paths = directory.rglob('*')
    dates = file_dates(paths)
    dates = dates.slice(start , end).filter(year)
    return dates

def load_df_max_date(path : strPath) -> int:
    """load dataframe from path"""
    from src.proj.db.io.dataframe import load_df_max_date
    return load_df_max_date(path)

def load_df_min_date(path : strPath) -> int:
    """load dataframe from path"""
    from src.proj.db.io.dataframe import load_df_min_date
    return load_df_min_date(path)
