"""Load dataframes from tar file and save dataframes to tar file."""

import pandas as pd
import polars as pl
import tarfile
import io

from pathlib import Path
from typing import Any , TypeVar

from src.proj.env import PATH
from src.proj.log import Logger
from src.proj.proj import Proj

from .core import PATH_TYPE , PD_MAPPER_TYPE
from .df_handler import dfHandler
from .df_io import dfIOHandler

__all__ = ['load_dfs_from_tar' , 'save_dfs_to_tar' , 'pack_files_to_tar' , 'unpack_files_from_tar']
T = TypeVar('T' , bound = pd.DataFrame | pl.DataFrame)

def load_tar(path : PATH_TYPE , mapper : PD_MAPPER_TYPE = None) -> dict[str , pd.DataFrame]:
    if not Path(path).exists():
        return {}
    dfs : dict[str , pd.DataFrame] = {}
    with tarfile.open(path, 'r') as tar: 
        try:
            for member in tar.getmembers():
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    dfs[member.name] = pd.DataFrame()
                else:
                    buffer = io.BytesIO(file_obj.read())
                    df = dfIOHandler.load_pandas(buffer , mapper = mapper)
                    dfs[member.name] = df
        except Exception as e:
            Logger.error(f'Error loading {path}: {e}')
            raise
    return dfs

def save_tar(dfs : dict[str , pd.DataFrame] , path : PATH_TYPE):
    """save multiple dataframes to tar file"""
    with tarfile.open(path, 'w') as tar:  # mode 'w' means not compress
        for name, df in dfs.items():
            tarinfo = tarfile.TarInfo(name)

            buffer = io.BytesIO()
            df = dfHandler.reset_index_pandas(df)
            if not isinstance(df.index , pd.RangeIndex):
                Logger.error(f'{df} is not a RangeIndex DataFrame')
                Logger.display(df , caption = 'Error saving DataFrame')
                raise ValueError(f'{df} is not a RangeIndex DataFrame')
            dfIOHandler.save_df(df , buffer)
            
            # get buffer size and reset pointer
            tarinfo.size = buffer.tell()
            buffer.seek(0)
            
            # add to tar (fully memory operation, no temporary file)
            tar.addfile(tarinfo, buffer)

def save_dfs_to_tar(dfs : dict[str , pd.DataFrame] , path : Path | str , *, overwrite = True , prefix = '' , indent = 1 , vb_level : Any = 1):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.suffix == '.tar' , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.unlink(missing_ok=True)
        save_tar(dfs , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def load_dfs_from_tar(path : str | Path , * , missing_ok = True , mapper : PD_MAPPER_TYPE = None) -> dict[str , pd.DataFrame]:
    """load multiple dataframes from tar file"""
    path = Path(path)
    if not path.exists():
        if missing_ok: 
            return {}
        else:
            raise FileNotFoundError(path)
    dfs = load_tar(path , mapper = dfHandler.wrapped_mapper(mapper))
    return dfs

def pack_files_to_tar(files : list[str | Path] , path : Path | str , *, overwrite = True , prefix = '' , indent = 1 , vb_level : Any = 1):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.suffix == '.tar' , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.unlink(missing_ok=True)
        with tarfile.open(path, 'a') as tar:  
            for file in files:
                tar.add(file , arcname = Path(file).relative_to(PATH.main))
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def unpack_files_from_tar(path : Path | str , target : Path | str , * , 
                          overwrite = False , indent = 1 , vb_level : Any = 1) -> None:
    """unpack files from tar file"""
    path = Path(path)
    target = Path(target)
    assert path.suffix == '.tar' , f'{path} is not a tar file'
    sub_vb_level = Proj.vb(vb_level) + 1
    with tarfile.open(path, 'r') as tar:  
        for member in tar.getmembers():
            target_path = target.joinpath(member.name)
            if not overwrite and target_path.exists():
                Logger.alert1(f"{target_path} already exists, skip unpacking" , indent = indent + 1 , vb_level = sub_vb_level)
            else:
                tar.extract(member, target)
                Logger.success(f"Unpacked {member.name} to {target}" , indent = indent + 1 , vb_level = sub_vb_level , italic = True)
    Logger.stdout(f"Unpacked {path} to {target}" , indent = indent , vb_level = vb_level , italic = True)
