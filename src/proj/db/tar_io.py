"""Load dataframes from tar file and save dataframes to tar file."""

import pandas as pd
import polars as pl
import tarfile
import json
import io

from pathlib import Path
from typing import Any , TypeVar

from src.proj.env import PATH , Proj
from src.proj.log import Logger
from src.proj.core import strPath

from .core import PD_MAPPER_TYPE
from .df_handler import dfHandler
from .df_io import dfIOHandler

__all__ = ['tar_suffixes' , 'load_tar_meta' , 'load_dfs_from_tar' , 'save_dfs_to_tar' , 'pack_files_to_tar' , 'unpack_files_from_tar']
T = TypeVar('T' , bound = pd.DataFrame | pl.DataFrame)
tar_suffixes = ['.tar' , '.tar.gz' , '.tar.bz2' , '.tar.xz' , '.tar.zst']

def load_tar(path : strPath , mapper : PD_MAPPER_TYPE = None) -> dict[str , pd.DataFrame]:
    """
    load dataframes from tar file , except meta.json
    Parameters:
        path: tar file path (support .tar, .tar.gz, .tar.bz2, .tar.xz)
        mapper: mapper function to execute on each dataframe
    Returns:
        dictionary of dataframes
    """
    if not Path(path).exists():
        return {}
    dfs : dict[str , pd.DataFrame] = {}
    with tarfile.open(path, 'r:*') as tar: 
        try:
            for member in tar.getmembers():
                if member.name == 'meta.json':
                    continue
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

def load_tar_meta(path : strPath) -> dict[str, Any]:
    """
    load meta.json from tar file
    
    Parameters:
        tar_path: tar file path (support .tar, .tar.gz, .tar.bz2, .tar.xz)
    
    Returns:
        dictionary of meta data; if file does not exist or meta.json does not exist, return empty dictionary
    """
    if not Path(path).exists():
        return {}
    
    with tarfile.open(path, 'r:*') as tar:
        try:
            member = tar.getmember('meta.json')
            file_obj = tar.extractfile(member)
            if file_obj is None:
                return {}
            with file_obj as f:
                return json.load(f)
        except KeyError:
            return {}

def save_tar(dfs : dict[str , pd.DataFrame] , path : strPath , meta : dict[str, Any] | None = None):
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

        if meta:
            meta_json_str = json.dumps(meta, indent=2, ensure_ascii=False)
            meta_bytes = meta_json_str.encode('utf-8')
            
            tarinfo = tarfile.TarInfo(name='meta.json')
            tarinfo.size = len(meta_bytes)
            tarinfo.mode = 0o644  # default permission
            tar.addfile(tarinfo, fileobj=io.BytesIO(meta_bytes))

def save_dfs_to_tar(dfs : dict[str , pd.DataFrame] , path : strPath , meta : dict[str, Any] | None = None , *, overwrite = True , prefix = '' , indent = 1 , vb_level : Any = 1):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.suffix == '.tar' , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.unlink(missing_ok=True)
        save_tar(dfs , path , meta = meta)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def load_dfs_from_tar(path : strPath , * , missing_ok = True , mapper : PD_MAPPER_TYPE = None) -> dict[str , pd.DataFrame]:
    """load multiple dataframes from tar file"""
    path = Path(path)
    if not path.exists():
        if missing_ok: 
            return {}
        else:
            raise FileNotFoundError(path)
    dfs = load_tar(path , mapper = dfHandler.wrapped_mapper(mapper))
    return dfs

def pack_files_to_tar(files : list[strPath] , path : strPath , *, overwrite = True , prefix = '' , indent = 1 , vb_level : Any = 1):
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

def unpack_files_from_tar(path : strPath , target : strPath , * , 
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
