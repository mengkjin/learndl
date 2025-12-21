import os
from dask.delayed import delayed
from dask.base import compute
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime , timedelta
from pathlib import Path
from typing import Any , Literal , Generator

from src.proj import MACHINE , PATH , Logger
from .code_mapper import secid_to_secid

__all__ = [
    'by_name' , 'by_date' , 'iter_db_srcs' , 'src_path' ,
    'save' , 'load' , 'load_multi' , 'rename' , 'path' , 'dates' , 'min_date' , 'max_date' ,
    'file_dates' , 'dir_dates' , 'save_df' , 'load_df' , 
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

_load_max_workers : int | Any = 40 if MACHINE.server else os.cpu_count()

_deprecated_db_by_name  : list[str] = ['information_js']
_deprecated_db_by_date  : list[str] = ['trade' , 'labels' , 'benchmark' , 'trade_js' , 'labels_js' , 'benchmark_js'] 

def _db_src_deprecated(i : int):
    """printing deprecated db_src information , i is the index of the db_src"""
    def wrapper(func):
        def inner(*args , **kwargs):
            db_src = args[i]
            if db_src in _deprecated_db_by_name or db_src in _deprecated_db_by_date:
                Logger.warning(f'at {func.__name__} , {db_src} will be deprecated soon, please update your code')
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

def save_df(df : pd.DataFrame | None , path : Path | str , overwrite = True , verbose = True , prefix = '  --> '):
    """save dataframe to path"""
    prefix = prefix or ''
    path = Path(path)
    if df is None or df.empty: 
        return False
    elif overwrite or not path.exists(): 
        status = 'Overwritten' if path.exists() else 'Saved to DB'
        path.parent.mkdir(parents=True , exist_ok=True)
        if SAVE_OPT_DB == 'feather':
            df.to_feather(path)
        else:
            df.to_parquet(path , engine='fastparquet')
        if verbose: 
            Logger.stdout(f'{prefix}{status}: {path}')
        return True
    else:
        status = 'File Exists'
        if verbose: 
            Logger.fail(f'{prefix}{status}: {path}')
        return False

def load_df(path : Path , raise_if_not_exist = False):
    """load dataframe from path"""
    if not path.exists():
        if raise_if_not_exist: 
            raise FileNotFoundError(path)
        else: 
            return pd.DataFrame()
    if SAVE_OPT_DB == 'feather':
        df = pd.read_feather(path)
    else:
        df = pd.read_parquet(path)
    df = _load_df_mapper(df)
    return df

def _load_df_mapper(df : pd.DataFrame):
    """map dataframe"""
    old_index = df.index.names if 'secid' in df.index.names else None
    if old_index is not None: 
        df = df.reset_index(drop = False)
    if 'secid' in df.columns:  
        df['secid'] = secid_to_secid(df['secid'])
    if old_index is not None: 
        df = df.set_index(old_index)
    if 'date' in df.index.names and 'date' in df.columns:
        df = df.reset_index('date' , drop = True)
    return df

def _load_df_multi(paths : dict , date_colname : str = 'date' , 
                  parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread'):
    """load dataframe from multiple paths"""
    if parallel is None: 
        parallel = _load_parallel
    reader : Any = pd.read_feather if SAVE_OPT_DB == 'feather' else pd.read_parquet
    paths = {d:p for d,p in paths.items() if p.exists()}
    if parallel is None or parallel == 'none':
        dfs = [reader(p).assign(**{date_colname:d}) for d,p in paths.items()]
    elif parallel == 'dask':
        ddfs = [delayed(reader)(p).assign(**{date_colname:d}) for d,p in paths.items()]
        dfs = compute(ddfs)[0]
    else:
        assert parallel == 'thread' or not MACHINE.is_windows, (parallel , MACHINE.system_name)
        max_workers = min(_load_max_workers , max(len(paths) // 5 , 1))
        PoolExecutor = ThreadPoolExecutor if parallel == 'thread' else ProcessPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(reader , p):d for d,p in paths.items()}
            dfs = [future.result().assign(**{date_colname:futures[future]}) for future in as_completed(futures)]

    if not dfs: 
        return pd.DataFrame()
    df = pd.concat([v for v in dfs if not v.empty])
    df = _load_df_mapper(df)
    return df

def _process_df(df : pd.DataFrame , date = None, date_colname = None , check_na_cols = False , 
               df_syntax : str | None = 'some df' , reset_index = True , ignored_fields = []):
    """process dataframe"""
    if date_colname and date is not None: 
        df[date_colname] = date

    if df_syntax:
        if df.empty:
            Logger.stdout(f'{df_syntax} is empty')
        else:
            na_cols : pd.Series | Any = df.isna().all()
            if na_cols.all():
                Logger.warn(f'{df_syntax} is all-NA')
            elif check_na_cols and na_cols.any():
                Logger.warn(f'{df_syntax} has columns [{str(df.columns[na_cols])}] all-NA')

    if reset_index and len(df.index.names) > 1 or df.index.name: 
        df = df.reset_index()
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

def min_date(db_src , db_key , use_alt = False):
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

def max_date(db_src , db_key , use_alt = False):
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
def save(df : pd.DataFrame | None , db_src , db_key , date = None , verbose = True , prefix : str | None = None):
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
    if df is not None and (len(df.index.names) > 1 or df.index.name): 
        df = df.reset_index()
    mark = save_df(df , _db_path(db_src , db_key , date , use_alt = False) , 
                   overwrite = True , verbose = verbose)
    return mark

# @_db_src_deprecated(0)
def load(db_src , db_key , date = None , date_colname = None , verbose = True , use_alt = False , 
         raise_if_not_exist = False , **kwargs) -> pd.DataFrame: 
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
    df_syntax = f'{db_src}/{db_key}/{date}' if verbose else None
    df = load_df(path , raise_if_not_exist = raise_if_not_exist)
    df = _process_df(df , date , date_colname , df_syntax = df_syntax , **kwargs)
    return df

# @_db_src_deprecated(0)
def load_multi(db_src , db_key , dates = None , start_dt = None , end_dt = None , 
                  date_colname = 'date' , 
                  verbose = True , use_alt = False , 
                  parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , **kwargs):
    """load multiple dates from database"""
    if db_src in DB_BY_NAME + EXPORT_BY_NAME:
        return load(db_src , db_key , dates = dates , start_dt = start_dt , end_dt = end_dt , 
                    date_colname = date_colname , verbose = verbose , use_alt = use_alt , 
                    parallel = parallel , **kwargs)
    if dates is None:
        assert start_dt is not None and end_dt is not None , f'start_dt and end_dt must be provided if dates is not provided'
        dates = _db_dates(db_src , db_key , start_dt , end_dt , use_alt = use_alt)
    paths : dict[int , Path] = {int(date):_db_path(db_src , db_key , date , use_alt = use_alt) for date in dates}
    df_syntax = f'{db_src}/{db_key}/multi-dates' if verbose else None
    df = _load_df_multi(paths , date_colname , parallel)
    df = _process_df(df , df_syntax = df_syntax , **kwargs)
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
