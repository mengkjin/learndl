from dask.delayed import delayed
from dask.base import compute
import numpy as np
import pandas as pd
import tarfile
import io

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime , timedelta
from pathlib import Path
from typing import Any , Literal , Generator , Callable

from src.proj.env import MACHINE , PATH
from src.proj.log import Logger
from .code_mapper import secid_to_secid

__all__ = [
    'by_name' , 'by_date' , 'iter_db_srcs' , 'src_path' ,
    'save' , 'load' , 'loads' , 'rename' , 'path' , 'dates' , 'min_date' , 'max_date' ,
    'file_dates' , 'dir_dates' , 'save_df' , 'save_dfs' , 'append_df' , 'load_df' , 'load_dfs' , 
    'load_df_max_date' , 'load_df_min_date' , 'load_dfs_from_tar' , 'save_dfs_to_tar' ,
    'block_path' , 'norm_path' ,
]

SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_MODEL: Literal['pt'] = 'pt'

DB_BY_NAME  : list[str] = ['information_js' , 'information_ts' , 'index_daily_ts']
DB_BY_DATE  : list[str] = ['models' , 'sellside' , 'exposure' ,
                           'trade_js' , 'labels_js' , 'benchmark_js' , 
                           'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 'holding_ts'
                           ]

EXPORT_BY_NAME : list[str] = ['market_factor' , 'factor_stats_daily' , 'factor_stats_weekly' , 'pooling_weight']
EXPORT_BY_DATE : list[str] = ['stock_factor' , 'model_prediction']
for name in EXPORT_BY_NAME + EXPORT_BY_DATE:
    assert name not in DB_BY_NAME + DB_BY_DATE , f'{name} must not in DB_BY_NAME and DB_BY_DATE'

_db_alternatives : dict[str , str] = {
    'trade_ts' : 'trade_js' ,
    'benchmark_ts' : 'benchmark_js'
}
_load_parallel : Literal['thread' , 'process' , 'dask' , 'none'] = 'thread'
# in Mac-Matthew , thread > dask > none > process

_load_max_workers : int | Any = 40 if MACHINE.server else MACHINE.cpu_count

_deprecated_db_by_name  : list[str] = ['information_js']
_deprecated_db_by_date  : list[str] = ['trade' , 'labels' , 'benchmark' , 'trade_js' , 'labels_js' , 'benchmark_js'] 

def _db_src_deprecated(i : int):
    """printing deprecated db_src information , i is the index of the db_src"""
    def wrapper(func):
        def inner(*args , **kwargs):
            db_src = args[i]
            if db_src in _deprecated_db_by_name or db_src in _deprecated_db_by_date:
                Logger.alert1(f'at {func.__name__} , {db_src} will be deprecated soon, please update your code')
            return func(*args , **kwargs)
        return inner
    return wrapper

def _today(offset = 0 , astype : Any = int):
    """get today's date"""
    d = datetime.today() + timedelta(days=offset)
    return astype(d.strftime('%Y%m%d'))

def _paths_to_dates(paths : list[Path] | Generator[Path, None, None]):
    """get dates from paths"""
    datestrs = [p.stem[-8:] for p in paths]
    dates = np.array([ds for ds in datestrs if ds.isdigit() and len(ds) == 8]).astype(int)
    dates.sort()
    return dates

def _df_loader(path : Path | io.BytesIO):
    """load dataframe from path"""
    try:
        if SAVE_OPT_DB == 'feather':
            return pd.read_feather(path)
        else:
            return pd.read_parquet(path , engine='fastparquet')
    except Exception as e:
        Logger.error(f'Error loading {path}: {e}')
        raise e

def _df_saver(df : pd.DataFrame , path : Path | io.BytesIO):
    """save dataframe to path"""
    try:
        if SAVE_OPT_DB == 'feather':
            df.to_feather(path)
        else:
            df.to_parquet(path , engine='fastparquet')
    except Exception as e:
        Logger.error(f'Error saving {path}: {e}')
        Logger.Display(df)
        raise e

def _tar_saver(dfs : dict[str , pd.DataFrame] , path : Path | str):
    """save multiple dataframes to tar file"""
    with tarfile.open(path, 'w') as tar:  # mode 'w' means not compress
        for name, df in dfs.items():
            tarinfo = tarfile.TarInfo(name)

            buffer = io.BytesIO()
            df = _reset_index(df)
            if not isinstance(df.index , pd.RangeIndex):
                Logger.error(f'{df} is not a RangeIndex DataFrame')
                Logger.Display(df)
                print(df.index)
                raise ValueError(f'{df} is not a RangeIndex DataFrame')
            _df_saver(df , buffer)
            
            # get buffer size and reset pointer
            tarinfo.size = buffer.tell()
            buffer.seek(0)
            
            # add to tar (fully memory operation, no temporary file)
            tar.addfile(tarinfo, buffer)

def _tar_loader(path : Path) -> dict[str , pd.DataFrame]:
    if not path.exists():
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
                    df = _df_loader(buffer)
                    dfs[member.name] = df
        except Exception as e:
            Logger.error(f'Error loading {path}: {e}')
            raise e
    return dfs

def by_name(db_src : str) -> bool:
    """whether the database is by name"""
    return db_src in DB_BY_NAME + EXPORT_BY_NAME

def by_date(db_src : str) -> bool:
    """whether the database is by date"""
    return db_src in DB_BY_DATE + EXPORT_BY_DATE

def iter_db_srcs() -> Generator[str, None, None]:
    """iterate over all database sources"""
    for db_src in DB_BY_NAME + DB_BY_DATE + EXPORT_BY_NAME + EXPORT_BY_DATE:
        yield db_src

def dir_dates(directory : Path , start_dt = None , end_dt = None , year = None):
    """get dates from directory"""
    paths = directory.rglob('*')
    dates = _paths_to_dates(paths)
    if end_dt   is not None: 
        dates = dates[dates <= (end_dt   if end_dt   > 0 else _today(end_dt))]
    if start_dt is not None: 
        dates = dates[dates >= (start_dt if start_dt > 0 else _today(start_dt))]
    if year is not None:     
        dates = dates[dates // 10000 == year]
    return dates

def file_dates(path : Path | list[Path] | tuple[Path] , startswith = '' , endswith = '') -> list:
    """get _db_path date from R environment"""
    if isinstance(path , (list,tuple)):
        return [d[0] for d in [file_dates(p , startswith , endswith) for p in path] if d]
    else:
        if not path.name.startswith(startswith): 
            return []
        if not path.name.endswith(endswith): 
            return []
        s = path.stem[-8:]
        return [int(s)] if s.isdigit() else []

def save_df(df : pd.DataFrame | None , path : Path | str , *, overwrite = True , prefix = '' , indent = 1 , vb_level : int = 1):
    """save dataframe to path"""
    if df is None or df.empty: 
        return False
    prefix = prefix or ''
    path = Path(path)
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.parent.mkdir(parents=True , exist_ok=True)
        _df_saver(df , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def save_dfs(dfs : dict[str , pd.DataFrame] , path : Path | str , * , overwrite = True , prefix = '' , indent = 1 , vb_level : int = 1):
    """save multiple dataframes to path (must be a directory or a tar file)"""
    if not dfs or all(df.empty for df in dfs.values()):
        return False
    prefix = prefix or ''
    path = Path(path)
    path.mkdir(parents=True , exist_ok=True)
    path_dfs = {path.joinpath(name):df for name , df in dfs.items() if not df.empty}
    if not overwrite and any(path_df.exists() for path_df in path_dfs):
        exists_paths = [path_df for path_df in path_dfs if path_df.exists()]
        Logger.alert1(f'{prefix} File Exists While not Overwriting: {path}' , indent = indent , vb_level = vb_level)
        Logger.alert1(f'{prefix} File Exists : {exists_paths}' , indent = indent + 1 , vb_level = vb_level)
        return False
    status : dict[str , int] = {'overwritten':0 , 'created':0}
    
    for df_path , df in path_dfs.items():
        status['overwritten'] += 1 if df_path.exists() else 0
        status['created'] += 1 if not df_path.exists() else 0
        _df_saver(df , df_path)
    Logger.stdout(f'{prefix} {status["overwritten"]} Overwritten , {status["created"]} Created: {path}' , indent = indent , vb_level = vb_level , italic = True)
    return True

def append_df(df : pd.DataFrame | None , path : Path | str , *, drop_duplicate_cols : list[str] | None = None , prefix = '' , indent = 1 , vb_level : int = 1):
    """append dataframe to path , can pass drop_duplicate_cols to drop duplicate columns"""
    path = Path(path)
    if df is None or df.empty: 
        return False
    elif not path.exists():
        return save_df(df , path , overwrite = True , prefix = prefix , indent = indent , vb_level = vb_level)
    else:
        status = 'Appended'
        df = pd.concat([load_df(path) , df])
        if drop_duplicate_cols:
            df = df.drop_duplicates(subset=drop_duplicate_cols , keep='last')
            status += f'with unique ({",".join(drop_duplicate_cols)})'
        _df_saver(df , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)

def load_df(path : Path , *, raise_if_not_exist = False):
    """load dataframe from path"""
    if not path.exists():
        if raise_if_not_exist: 
            raise FileNotFoundError(path)
        else: 
            return pd.DataFrame()
    df = _df_loader(path)
    df = _load_df_mapper(df)
    return df

def load_df_max_date(path : Path , date_colname : str = 'date') -> int:
    """load dataframe from path"""
    if not path.exists() or (df := load_df(path)).empty:
        return 19000101
    else:
        return int(max(df[date_colname]))

def load_df_min_date(path : Path , date_colname : str = 'date') -> int:
    """load dataframe from path"""
    if not path.exists() or (df := load_df(path)).empty:
        return 99991231
    else:
        return int(min(df[date_colname]))

def load_dfs(paths : dict | list[Path] , * ,  key_column : str | None = 'date' , 
             parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , 
             mapper : Callable[[pd.DataFrame], pd.DataFrame] | None = None):
    """
    load dataframe from multiple paths
    Parameters
    ----------
    paths : dict[int, Path]
        paths to load , key is date
    key_column : str | None
        key column name , if None, use date column
    parallel : Literal['thread' , 'process' , 'dask' , 'none']
        parallel mode
    mapper : Callable[[pd.DataFrame], pd.DataFrame]
        mapper function to execute on each dataframe
    """
    if parallel is None: 
        parallel = _load_parallel

    if mapper is None:
        def loader(p : Path):
            return _df_loader(p)
    else:
        def loader(p : Path):
            return mapper(_df_loader(p))
    
    if isinstance(paths , dict):
        paths = {d:p for d,p in paths.items() if p.exists()}
        assign_col = key_column if key_column else 'empty_column'
    else:
        paths = {i:p for i,p in enumerate(paths) if p.exists()}
        assign_col = 'empty_column'

    if not paths:
        return pd.DataFrame()
    
    if parallel is None or parallel == 'none':
        dfs = [loader(p).assign(**{assign_col:d}) for d,p in paths.items()]
    elif parallel == 'dask':
        ddfs = [delayed(loader)(p).assign(**{assign_col:d}) for d,p in paths.items()]
        dfs = compute(ddfs)[0]
    else:
        assert parallel == 'thread' or not MACHINE.is_windows, (parallel , MACHINE.system_name)
        max_workers = min(_load_max_workers , max(len(paths) // 5 , 1))
        PoolExecutor = ThreadPoolExecutor if parallel == 'thread' else ProcessPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(loader , p):d for d,p in paths.items()}
            dfs = [future.result().assign(**{assign_col:futures[future]}) for future in as_completed(futures)]

    df = _load_df_mapper(pd.concat([v for v in dfs if not v.empty]))
    if key_column is None:
        df = df.drop(columns = 'empty_column')
    return df

def save_dfs_to_tar(dfs : dict[str , pd.DataFrame] , path : Path | str , *, overwrite = True , prefix = '' , indent = 1 , vb_level : int = 1):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.suffix == '.tar' , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.unlink(missing_ok=True)
        _tar_saver(dfs , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def load_dfs_from_tar(path : Path , * , raise_if_not_exist = False) -> dict[str , pd.DataFrame]:
    """load multiple dataframes from tar file"""
    path = Path(path)
    if not path.exists():
        if raise_if_not_exist: 
            raise FileNotFoundError(path)
        else: 
            return {}
    dfs = _tar_loader(path)
    for key , df in dfs.items():
        dfs[key] = _load_df_mapper(df)
    return dfs

def _reset_index(df : pd.DataFrame | Any , reset = True):
    """reset index which are not None"""
    if not reset or df is None or df.empty:
        return df
    old_index = [index for index in df.index.names if index]
    df = df.reset_index(old_index , drop = False)
    if isinstance(df.index , pd.RangeIndex):
        df = df.reset_index(drop = True)
    return df

def _load_df_mapper(df : pd.DataFrame):
    if 'date' in df.index.names and 'date' in df.columns:
        df = df.reset_index('date' , drop = True)
    old_index = [idx for idx in df.index.names if idx]
    df = _reset_index(df)
    if 'secid' in df.columns:  
        df['secid'] = secid_to_secid(df['secid'])
    if old_index: 
        df = df.set_index(old_index)
    return df

def _process_df(df : pd.DataFrame , date = None, date_colname = None , check_na_cols = False , 
               df_syntax : str = 'some df' , reset_index = True , ignored_fields = [] , indent = 1 , vb_level : int = 1):
    """process dataframe"""
    if date_colname and date is not None: 
        df[date_colname] = date

    if df.empty:
        Logger.alert1(f'{df_syntax} is empty' , indent = indent , vb_level = vb_level)
    else:
        na_cols : pd.Series | Any = df.isna().all()
        if na_cols.all():
            Logger.alert1(f'{df_syntax} is all-NA' , indent = indent)
        elif check_na_cols and na_cols.any():
            Logger.alert1(f'{df_syntax} has columns [{str(df.columns[na_cols])}] all-NA' , indent = indent)

    df = _reset_index(df , reset_index)
    if ignored_fields: 
        df = df.drop(columns=ignored_fields , errors='ignore')
    return df

def _db_parent(db_src : str , db_key : str | None = None) -> Path:
    """get database parent _db_path"""
    if db_src in DB_BY_NAME + DB_BY_DATE:
        parent = PATH.database.joinpath(f'DB_{db_src}')
    elif db_src in ['pred' , 'factor']:
        parent = getattr(PATH , db_src)
    elif db_src in EXPORT_BY_NAME + EXPORT_BY_DATE:
        parent = PATH.export.joinpath(db_src)
    else:
        raise ValueError(f'{db_src} not in {DB_BY_NAME} / {DB_BY_DATE} / {EXPORT_BY_NAME} / {EXPORT_BY_DATE} / pred / factor')
    if db_key is None or db_src in DB_BY_NAME + EXPORT_BY_NAME:
        return parent
    else:
        return parent.joinpath(db_key)

def src_path(db_src : str) -> Path:
    """get database source path"""
    return _db_parent(db_src)

def _db_path(db_src , db_key , date = None , use_alt = False) -> Path:
    """
    Get path of database
    Parameters
    ----------
    db_src: str
        database source name , or factor or pred
    db_key: str
        database key , or factor name or pred name
    date: int, default None
        date to be saved, if the db is by date, date is required
    """
    parent = _db_parent(db_src , db_key)
    if db_src in DB_BY_NAME + EXPORT_BY_NAME:
        new_path = parent.joinpath(f'{db_key}.{SAVE_OPT_DB}')
    else:
        assert date is not None , f'{db_src} use date type but date is None'
        new_path = parent.joinpath(str(int(date) // 10000) , f'{db_key}.{str(date)}.{SAVE_OPT_DB}')
    if not new_path.exists() and db_src in _db_alternatives and use_alt:
        alt_path = _db_path(_db_alternatives[db_src] , db_key , date , use_alt = False)
        if alt_path.exists(): 
            new_path = alt_path
    return new_path

def _db_dates(db_src , db_key , start_dt = None , end_dt = None , year = None , use_alt = False):
    """get dates from any database data"""
    path = _db_parent(db_src , db_key)
    dates = dir_dates(path , start_dt , end_dt , year)
    if db_src in _db_alternatives and use_alt:
        alt_src   = _db_alternatives[db_src]
        alt_path  = _db_parent(alt_src , db_key)
        alt_dates = np.setdiff1d(dir_dates(alt_path , start_dt , end_dt , year) , dates)
        dates = np.concatenate([alt_dates , dates])
    return dates

def min_date(db_src , db_key , *, use_alt = False):
    """get minimum date from any database data"""
    directory = _db_parent(db_src , db_key)
    years = [int(y.stem) for y in directory.iterdir() if y.is_dir()] if directory.exists() else []
    if years: 
        paths = [p for p in directory.joinpath(str(min(years))).iterdir()]
        dates = _paths_to_dates(paths)
        mdate = min(dates) if len(dates) else 99991231
    else:
        mdate = 99991231
    if db_src in _db_alternatives and use_alt:
        alt_src   = _db_alternatives[db_src]
        directory = _db_parent(alt_src , db_key)
        years = [int(y.stem) for y in directory.iterdir() if y.is_dir()] if directory.exists() else []
        if years: 
            paths = [p for p in directory.joinpath(str(min(years))).iterdir()]
            dates = _paths_to_dates(paths)
            mdate = min(mdate , min(dates) if len(dates) else 99991231)
    return int(mdate)

def max_date(db_src , db_key , *, use_alt = False):
    """get maximum date from any database data"""
    directory = _db_parent(db_src , db_key)
    years = [int(y.stem) for y in directory.iterdir() if y.is_dir()] if directory.exists() else []
    if years: 
        paths = [p for p in directory.joinpath(str(max(years))).iterdir()]
        dates = _paths_to_dates(paths)
        mdate = max(dates) if len(dates) else 0
    else:
        mdate = 0
    if db_src in _db_alternatives and use_alt:
        alt_src   = _db_alternatives[db_src]
        directory = _db_parent(alt_src , db_key)
        years = [int(y.stem) for y in directory.iterdir() if y.is_dir()] if directory.exists() else []
        if years: 
            paths = [p for p in directory.joinpath(str(max(years))).iterdir()]
            dates = _paths_to_dates(paths)
            mdate = max(mdate , max(dates) if len(dates) else 0)
    return int(mdate)

# @_db_src_deprecated(1)
def save(df : pd.DataFrame | None , db_src : str , db_key : str , date = None , *, overwrite = True , indent = 1 , vb_level : int = 1 , reason : str = ''):
    '''
    Save data to database
    Parameters  
    ----------
    df: pd.DataFrame | None
        data to be saved
    db_src: str
        database source name , or export source name
    db_key: str
        database key , or export key name
    date: int, default None
        date to be saved, if the db is by date, date is required
    '''
    df = _reset_index(df , reset = True)
    mark = save_df(df , _db_path(db_src , db_key , date , use_alt = False) , 
                   overwrite = overwrite , prefix = f'{db_src.title()} {reason}' if reason else db_key , indent = indent , vb_level = vb_level)
    return mark

# @_db_src_deprecated(0)
def load(db_src , db_key , date = None , *, 
         date_colname = None , use_alt = False , 
         raise_if_not_exist = False , indent = 1 , vb_level : int = 1 , **kwargs) -> pd.DataFrame: 
    '''
    Load data from database
    Parameters
    ----------
    db_src: str
        database source name , or export source name (etc. pred , factor , market_factor , factor_stats_daily , factor_stats_weekly)
    db_key: str
        database key , or export key name
    date: int, default None
        date to be loaded , if the db is by date , date is required
    date_colname: str, default None
        date column name , if submitted , the date will be assigned to this column
    silent: default False
        if True, no message will be printed
    kwargs: kwargs for process_df
        raise_if_not_exist: bool, default False
            if True, raise FileNotFoundError if the file does not exist
        ignored_fields: list, default []
            fields to be dropped , consider ['wind_id' , 'stockcode' , 'ticker' , 's_info_windcode' , 'code']
        reset_index: bool, default True
            if True, reset index (no drop index)
    '''
    path = _db_path(db_src , db_key , date , use_alt = use_alt)
    df = load_df(path , raise_if_not_exist = raise_if_not_exist)
    df = _process_df(df , date , date_colname , df_syntax = f'{db_src}/{db_key}/{date}' , indent = indent , vb_level = vb_level , **kwargs)
    return df

# @_db_src_deprecated(0)
def loads(db_src , db_key , dates = None , start_dt = None , end_dt = None , *,
               date_colname = 'date' , use_alt = False , 
               parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , indent = 1 , vb_level : int = 1 , **kwargs):
    """load multiple dates from database"""
    if db_src in DB_BY_NAME + EXPORT_BY_NAME:
        return load(db_src , db_key , dates = dates , start_dt = start_dt , end_dt = end_dt , 
                    date_colname = date_colname , use_alt = use_alt , 
                    parallel = parallel , indent = indent , vb_level = vb_level , **kwargs)
    if dates is None:
        assert start_dt is not None or end_dt is not None , f'start_dt or end_dt must be provided if dates is not provided'
        dates = _db_dates(db_src , db_key , start_dt , end_dt , use_alt = use_alt)
    paths : dict[int , Path] = {int(date):_db_path(db_src , db_key , date , use_alt = use_alt) for date in dates}
    df = load_dfs(paths , key_column = date_colname , parallel = parallel)
    df = _process_df(df , df_syntax = f'{db_src}/{db_key}/multi-dates' , indent = indent , vb_level = vb_level , **kwargs)
    return df

def rename(db_src , db_key , new_db_key):
    """rename database from db_key to new_db_key"""
    assert new_db_key not in PATH.list_files(_db_parent(db_src , db_key)) , f'{new_db_key} already exists'
    if db_src in DB_BY_NAME:
        old_path = _db_path(db_src , db_key)
        new_path = _db_path(db_src , new_db_key)
        old_path.rename(new_path)
    else:
        for date in _db_dates(db_src , db_key):
            old_path = _db_path(db_src , db_key , date)
            new_path = _db_path(db_src , new_db_key , date)
            new_path.parent.mkdir(parents=True , exist_ok=True)
            old_path.rename(new_path)
        root = _db_parent(db_src , db_key)
        [d.rmdir() for d in root.iterdir() if d.is_dir()]
        root.rmdir()


def path(db_src , db_key , date = None , use_alt = False) -> Path:
    """
    Get path of database
    Parameters
    ----------
    db_src: str
        database source name , or export source name (etc. pred , factor , market_factor , factor_stats_daily , factor_stats_weekly)
    db_key: str
        database key , or export key name
    date: int, default None
        date to be saved, if the db is by date, date is required
    """
    return _db_path(db_src , db_key , date , use_alt = use_alt)

def dates(db_src , db_key , start_dt = None , end_dt = None , year = None , use_alt = False):
    """get dates from any database data"""
    return _db_dates(db_src , db_key , start_dt , end_dt , year , use_alt)

def block_path(name : str) -> Path:
    return PATH.block.joinpath(f'{name}.{SAVE_OPT_BLK}')

def norm_path(name : str) -> Path:
    return PATH.norm.joinpath(f'{name}.{SAVE_OPT_NORM}')
