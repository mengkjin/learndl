"""Load dataframes from tar file and save dataframes to tar file."""
from __future__ import annotations
import pandas as pd
import tarfile
import json
import io

from pathlib import Path
from typing import Any , TYPE_CHECKING
from collections.abc import Mapping

from src.proj.env import PATH , Proj
from src.proj.log import Logger
from src.proj.core import strPath , strPaths , lit

from src.proj.db.basic import dfHandler , TAR_SUFFIXES
from .dataframe import dfIOHandler

if TYPE_CHECKING:
    import polars as pl
    from src.proj.db.io.dataframe import PD_MAPPER_TYPE

__all__ = ['load_tar_meta' , 'load_dfs_from_tar' , 'save_dfs_to_tar' , 'pack_files_to_tar' , 'unpack_files_from_tar']

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

def save_tar(dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath , meta : dict[str, Any] | None = None):
    """save multiple dataframes to tar file"""
    with tarfile.open(path, 'w') as tar:  # mode 'w' means not compress
        for name, df in dfs.items():
            tarinfo = tarfile.TarInfo(name)
            if not isinstance(df , pd.DataFrame):
                df = df.to_pandas()
            buffer = io.BytesIO()
            df = dfHandler.reset_index_pandas(df)
            if df.index.__class__.__qualname__ != 'RangeIndex':
                Logger.error(f'{df} is not a RangeIndex DataFrame')
                Logger.display(df , title = 'Error saving DataFrame')
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

def save_dfs_to_tar(
    dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath , 
    meta : dict[str, Any] | None = None , *, overwrite = True , prefix : str | None = None , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 1
):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.name.endswith(TAR_SUFFIXES) , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.unlink(missing_ok=True)
        save_tar(dfs , path , meta = meta)
        Logger.stdout(f'{prefix}{status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix}{status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def load_dfs_from_tar(path : strPath , * , missing_ok = True , mapper : PD_MAPPER_TYPE = None , **kwargs) -> dict[str , pd.DataFrame]:
    """load multiple dataframes from tar file"""
    path = Path(path)
    if not path.exists():
        if missing_ok: 
            return {}
        else:
            raise FileNotFoundError(path)
    dfs = load_tar(path , mapper = dfHandler.wrapped_mapper(mapper))
    return dfs

def pack_files_to_tar(
    files : strPaths , path : strPath , *, overwrite = True , prefix : str | None = None , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 1 , **kwargs
):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.name.endswith(TAR_SUFFIXES) , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        if isinstance(files , dict | Mapping):
            files = list(files.values())
        with tarfile.open(path, 'a') as tar:  
            for file in files:
                tar.add(file , arcname = PATH.relative(file))
        Logger.footnote(f'{prefix}{status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix}{status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def pack_dir_to_tar(
    source_path : strPath , path : strPath , * ,
    prefix : str | None = None , overwrite = False , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 3 , **kwargs
):
    source_path = Path(source_path)
    path = Path(path)
    assert source_path.exists() and source_path.is_dir() and any(source_path.iterdir()) , f'{source_path} does not exist or is empty'
    assert path.name.endswith(TAR_SUFFIXES) , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        with tarfile.open(path, 'w:gz') as tar:
            for file in source_path.rglob('*'):
                tar.add(file, arcname=PATH.relative(file))
        Logger.footnote(f'{prefix}{status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix}{status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def unpack_files_from_tar(
    path : strPath , target : strPath , * , 
    overwrite = False , indent : int = 1 , vb_level : lit.VerbosityLevel = 1 , **kwargs
) -> None:
    """unpack files from tar file"""
    path = Path(path)
    target = Path(target)
    assert path.name.endswith(TAR_SUFFIXES) , f'{path} is not a tar file'
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

def packing(
    source_path : strPath | strPaths , target_path : strPath , * ,
    prefix : str | None = None , overwrite = False , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 3 , **kwargs
):
    if isinstance(source_path , strPath) and Path(source_path).is_dir():
        return pack_dir_to_tar(
            source_path, target_path, prefix = prefix, overwrite = overwrite, 
            indent = indent, vb_level = vb_level, **kwargs
        )
    else:
        if isinstance(source_path , strPath):
            source_path = [source_path]
        return pack_files_to_tar(
            source_path, target_path, prefix = prefix, overwrite = overwrite, 
            indent = indent, vb_level = vb_level, **kwargs
        )