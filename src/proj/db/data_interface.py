from dask.delayed import delayed
from dask.base import compute
import numpy as np
import pandas as pd
import tarfile
import io

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any , Literal , Generator , Callable

from src.proj.env import MACHINE , PATH
from src.proj.log import Logger
from src.proj.proj import Proj
from .code_mapper import secid_to_secid

__all__ = [
    'DBPath' ,
    'save' , 'load' , 'loads' , 'rename' , 'path' , 'dates' , 'min_date' , 'max_date' ,
    'file_dates' , 'dir_dates' , 'save_df' , 'save_dfs' , 'append_df' , 'load_df' , 'load_dfs' ,  'load_dfs_seperately' , 
    'load_df_max_date' , 'load_df_min_date' , 'load_dfs_from_tar' , 'save_dfs_to_tar' , 
    'pack_files_to_tar' , 'unpack_files_from_tar'
]

DATAFRAME_SUFFIX   : Literal['feather' , 'parquet'] = 'feather'
DEFAULT_PARALLEL_METHOD : Literal['thread' , 'process' , 'dask' , 'none'] = 'thread'   # in Mac-Matthew , thread > dask > none > process

SRC_ALTERNATIVES : dict[str , list[str]] = {
    'trade_ts' : ['trade_js'] ,
    'benchmark_ts' : ['benchmark_js']
}
DB_BY_NAME  : list[str] = [
    'information_js' , 'information_ts' , 'index_daily_ts' ,  'index_daily_custom' , 'market_daily']
DB_BY_DATE  : list[str] = [
    'models' , 'sellside' , 'exposure' , 'trade_js' , 'labels_js' , 'benchmark_js' , 
    'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 'holding_ts' ,
    'crawler'
]
EXPORT_BY_NAME : list[str] = ['market_factor' , 'factor_stats_daily' , 'factor_stats_weekly' , 'pooling_weight']
EXPORT_BY_DATE : list[str] = ['stock_factor' , 'model_prediction' , 'universe']
for name in EXPORT_BY_NAME + EXPORT_BY_DATE:
    assert name not in DB_BY_NAME + DB_BY_DATE , f'{name} must not in DB_BY_NAME and DB_BY_DATE'


def paths_to_dates(paths : list[Path] | Generator[Path, None, None]):
    """get dates from paths"""
    datestrs = [p.stem[-8:] for p in paths]
    dates = np.array([ds for ds in datestrs if ds.isdigit() and len(ds) == 8] , dtype = int)
    dates.sort()
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

def dir_dates(directory : Path , start = None , end = None , year = None):
    """get dates from directory"""
    paths = directory.rglob('*')
    dates = paths_to_dates(paths)
    if end   is not None: 
        dates = dates[dates <= end]
    if start is not None: 
        dates = dates[dates >= start]
    if year is not None:
        dates = dates[dates // 10000 == year]
    return dates

class FileIOHandler:
    """File IO operations handler"""
    @classmethod
    def load_df(cls , path : Path | io.BytesIO , * , mapper : Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> pd.DataFrame:
        """load dataframe from path"""
        try:
            if DATAFRAME_SUFFIX == 'feather':
                df = pd.read_feather(path)
            else:
                df = pd.read_parquet(path , engine='fastparquet')
        except Exception as e:
            Logger.error(f'Error loading {path}: {e}')
            raise
        if mapper is not None:
            df = mapper(df)
        return df
    
    @classmethod
    def save_df(cls , df : pd.DataFrame , path : Path | io.BytesIO):
        """save dataframe to path"""
        try:
            if DATAFRAME_SUFFIX == 'feather':
                df.to_feather(path)
            else:
                df.to_parquet(path , engine='fastparquet')
        except Exception as e:
            Logger.error(f'Error saving {path}: {e}')
            Logger.display(df , caption = 'Error saving DataFrame')
            raise

    @classmethod
    def load_tar(cls , path : Path , mapper : Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> dict[str , pd.DataFrame]:
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
                        df = cls.load_df(buffer , mapper = mapper)
                        dfs[member.name] = df
            except Exception as e:
                Logger.error(f'Error loading {path}: {e}')
                raise
        return dfs

    @classmethod
    def save_tar(cls , dfs : dict[str , pd.DataFrame] , path : Path | str):
        """save multiple dataframes to tar file"""
        with tarfile.open(path, 'w') as tar:  # mode 'w' means not compress
            for name, df in dfs.items():
                tarinfo = tarfile.TarInfo(name)

                buffer = io.BytesIO()
                df = DFProcessor.reset_index(df)
                if not isinstance(df.index , pd.RangeIndex):
                    Logger.error(f'{df} is not a RangeIndex DataFrame')
                    Logger.display(df , caption = 'Error saving DataFrame')
                    raise ValueError(f'{df} is not a RangeIndex DataFrame')
                cls.save_df(df , buffer)
                
                # get buffer size and reset pointer
                tarinfo.size = buffer.tell()
                buffer.seek(0)
                
                # add to tar (fully memory operation, no temporary file)
                tar.addfile(tarinfo, buffer)

    @classmethod
    def parallel_load_df(cls , path_dict : dict[int | Any, Path] , * , parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , 
                      mapper : Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> dict[int | Any, pd.DataFrame]:
        if parallel is None: 
            parallel = DEFAULT_PARALLEL_METHOD
        
        def loader(p : Path):
            return cls.load_df(p , mapper = mapper)

        paths = {d:p for d,p in path_dict.items() if p.exists()}
        if not paths:
            return {}
        if parallel is None or parallel == 'none':
            dfs = {d:loader(p) for d,p in paths.items() if not loader(p).empty}
        elif parallel == 'dask':
            ddfs = [delayed(loader)(p) for d,p in paths.items()]
            dfs = {d:df for d,df in zip(paths.keys() , compute(ddfs)[0])}
        else:
            assert parallel == 'thread' or not MACHINE.is_windows, (parallel , MACHINE.system_name)
            max_workers = min(MACHINE.max_workers , max(len(paths) // 5 , 1))
            PoolExecutor = ThreadPoolExecutor if parallel == 'thread' else ProcessPoolExecutor
            with PoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(loader , p):d for d,p in paths.items()}
                dfs = {futures[future]:future.result() for future in as_completed(futures)}
        return dfs

class DFProcessor:
    @classmethod
    def reset_index(cls , df : pd.DataFrame | Any , reset = True):
        """reset index which are not None"""
        if not reset or df is None or df.empty:
            return df
        old_index = [index for index in df.index.names if index]
        df = df.reset_index(old_index , drop = False)
        if isinstance(df.index , pd.RangeIndex):
            df = df.reset_index(drop = True)
        return df

    @classmethod
    def load_mapper(cls , df : pd.DataFrame):
        if 'date' in df.index.names and 'date' in df.columns:
            df = df.reset_index('date' , drop = True)
        old_index = [idx for idx in df.index.names if idx]
        df = cls.reset_index(df)
        if 'secid' in df.columns:  
            df['secid'] = secid_to_secid(df['secid'])
        if old_index: 
            df = df.set_index(old_index)
        return df

    @classmethod
    def load_process(cls , df : pd.DataFrame , date = None, date_colname = None , check_na_cols = False , 
                     df_syntax : str = 'some df' , reset_index = True , ignored_fields = [] , indent = 1 , vb_level : Any = 'max'):
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

        df = cls.reset_index(df , reset_index)
        if ignored_fields: 
            df = df.drop(columns=ignored_fields , errors='ignore')
        return df
    
class DBPath:
    """DB Path structure of db_src and db_key"""
    src_alternatives = SRC_ALTERNATIVES
    db_by_name = DB_BY_NAME
    db_by_date = DB_BY_DATE
    export_by_name = EXPORT_BY_NAME
    export_by_date = EXPORT_BY_DATE

    instance_cache : dict[str , 'DBPath'] = {}

    def __new__(cls , db_src : str , db_key : str):
        if f'{db_src}/{db_key}' not in cls.instance_cache:
            cls.instance_cache[f'{db_src}/{db_key}'] = super().__new__(cls)
        return cls.instance_cache[f'{db_src}/{db_key}']

    def __init__(self , db_src : str , db_key : str):
        self.src = db_src
        self.key = db_key

    def __repr__(self):
        return f'{self.src}/{self.key}'

    @classmethod
    def iter_srcs(cls) -> Generator[str, None, None]:
        """iterate over all database sources"""
        for db_src in cls.db_by_name + cls.db_by_date + cls.export_by_name + cls.export_by_date:
            yield db_src

    @classmethod
    def ByName(cls , db_src : str) -> bool:
        """whether the database is by name"""
        return db_src in cls.db_by_name + cls.export_by_name

    @classmethod
    def ByDate(cls , db_src : str) -> bool:
        """whether the database is by date"""
        return db_src in cls.db_by_date + cls.export_by_date

    @classmethod
    def Parent(cls , db_src : str , db_key : str | None = None) -> Path:
        """get database parent _db_path"""
        if db_src in cls.db_by_name + cls.db_by_date:
            parent = PATH.database.joinpath(f'DB_{db_src}')
        elif db_src in ['pred' , 'factor']:
            parent = getattr(PATH , db_src)
        elif db_src in cls.export_by_name + cls.export_by_date:
            parent = PATH.export.joinpath(db_src)
        else:
            raise ValueError(f'{db_src} not in {cls.db_by_name} / {cls.db_by_date} / {cls.export_by_name} / {cls.export_by_date} / pred / factor')
        if db_key is None or db_src in cls.db_by_name + cls.export_by_name:
            return parent
        else:
            return parent.joinpath(db_key)

    @classmethod
    def PathExact(cls , db_src : str , db_key : str , date : int | None = None) -> Path:
        """get exact path of database"""
        if db_src in cls.db_by_name + cls.export_by_name:
            return cls(db_src , db_key).parent.joinpath(f'{db_key}.{DATAFRAME_SUFFIX}')
        else:
            assert date is not None , f'{db_src} use date type but date is None'
            return cls(db_src , db_key).parent.joinpath(str(int(date) // 10000) , f'{db_key}.{str(date)}.{DATAFRAME_SUFFIX}')

    @property
    def parent(self) -> Path:
        """get database parent _db_path"""
        return self.Parent(self.src , self.key)

    @property
    def by_name(self) -> bool:
        """whether the database is by name"""
        return self.ByName(self.src)

    @property
    def by_date(self) -> bool:
        """whether the database is by date"""
        return self.ByDate(self.src)

    def years(self) -> list[int]:
        """get years from database"""
        directory = self.parent
        return [int(y.stem) for y in directory.iterdir() if y.is_dir() and any(y.iterdir())] if directory.exists() else []

    def dates(self , start = None , end = None , year = None , * , use_alt = False) -> np.ndarray:
        """get dates from any database data"""
        if use_alt:
            candidates = [self] + self.alternatives()
            dates = np.unique(np.concatenate([db_path.dates(start , end , year , use_alt = False) for db_path in candidates]))
        else:
            directory = self.parent.joinpath(str(year)) if year else self.parent
            dates = dir_dates(directory , start , end)
        return dates
    
    def min_date(self , * , use_alt = False):
        """get minimum date from any database data"""
        if use_alt:
            candidates = [self] + self.alternatives()
            return min(db_path.min_date(use_alt = False) for db_path in candidates)
        else:
            directory = self.parent
            years = self.years()
            if years: 
                dates = paths_to_dates(directory.joinpath(str(min(years))).iterdir())
                mdate = min(dates) if len(dates) else 99991231
            else:
                mdate = 99991231
            return int(mdate)

    def max_date(self , * , use_alt = False):
        """get maximum date from any database data"""
        if use_alt:
            candidates = [self] + self.alternatives()
            return max(db_path.max_date(use_alt = False) for db_path in candidates)
        else:
            directory = self.parent
            years = self.years()
            if years: 
                dates = paths_to_dates(directory.joinpath(str(max(years))).iterdir())
                mdate = max(dates) if len(dates) else 0
            else:
                mdate = 0
            return int(mdate)

    def date_closest(self , date : int | None , * , within_years : int = 1) -> int | None:
        """get closest date from database"""
        if date is None:
            return None
        year = int(date) // 10000
        for minus_year in range(within_years + 1):
            dates = self.dates(end = date , year = year - minus_year)
            if len(dates) > 0:
                return max(dates)
        return None

    def alternatives(self) -> list['DBPath']:
        """get alternatives of database"""
        if self.src in self.src_alternatives:
            return [DBPath(alt_src , self.key) for alt_src in self.src_alternatives[self.src]]
        return []

    def path_exact(self , date = None) -> Path:
        """get exact path of database"""
        return self.PathExact(self.src , self.key , date)

    def path_closest(self , date = None) -> Path:
        """get closest path of database"""
        if self.by_name:
            path = self.path_exact()
        else:
            assert date is not None , f'{self.src} use date type but date is None'
            date = self.date_closest(date) or date
            path = self.path_exact(date)
        return path

    def path(self , date : int | None = None , use_alt = False , closest = False , indent = 1 , vb_level : Any = 'max') -> Path:
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
        path = self.path_exact(date)
        if path.exists():
            return path

        candidates = self.alternatives() if use_alt else []
        for db_path in candidates:
            if (alt_path := db_path.path(date)).exists():
                Logger.stdout(f'{self} use alternative path: {alt_path}' , indent = indent , vb_level = vb_level , italic = True)
                return alt_path
            
        if closest:
            all_candidates = [self] + candidates
            closest_dates = [db_path.date_closest(date) or -1 for db_path in all_candidates]
            idx = np.argmax(closest_dates)
            alt_path = all_candidates[idx].path_exact(closest_dates[idx])
            if alt_path.exists():
                Logger.stdout(f'{self} use closest path: {alt_path}' , indent = indent , vb_level = vb_level , italic = True)
                return alt_path

        return path

    def rename(self , new_db_key : str):
        """rename database from db_key to new_db_key"""
        assert new_db_key not in PATH.list_files(self.parent.parent) , f'{new_db_key} already exists in {self.parent}'
        if self.by_name:
            old_path = self.path_exact()
            new_path = self.PathExact(self.src , new_db_key)
            old_path.rename(new_path)
        else:
            for date in self.dates():
                old_path = self.path_exact(date)
                new_path = self.PathExact(self.src , new_db_key , date)
                new_path.parent.mkdir(parents=True , exist_ok=True)
                old_path.rename(new_path)
            [d.rmdir() for d in self.parent.iterdir() if d.is_dir()]
            self.parent.rmdir()

def save_df(df : pd.DataFrame | None , path : Path | str , *, overwrite = True , prefix = '' , empty_ok = False , indent = 1 , vb_level : Any = 1):
    """save dataframe to path"""
    if df is None or (not empty_ok and df.empty): 
        return False
    prefix = prefix or ''
    path = Path(path)
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.parent.mkdir(parents=True , exist_ok=True)
        FileIOHandler.save_df(df , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def save_dfs(dfs : dict[str , pd.DataFrame] , path : Path | str , * , overwrite = True , prefix = '' , indent = 1 , vb_level : Any = 1):
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
        FileIOHandler.save_df(df , df_path)
    Logger.stdout(f'{prefix} {status["overwritten"]} Overwritten , {status["created"]} Created: {path}' , indent = indent , vb_level = vb_level , italic = True)
    return True

def append_df(df : pd.DataFrame | None , path : Path | str , *, drop_duplicate_cols : list[str] | None = None , prefix = '' , indent = 1 , vb_level : Any = 1):
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
        FileIOHandler.save_df(df , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)

def load_df(path : Path | str , *, raise_if_not_exist = False):
    """load dataframe from path"""
    path = Path(path)
    if not path.exists():
        if raise_if_not_exist: 
            raise FileNotFoundError(path)
        else: 
            return pd.DataFrame()
    df = FileIOHandler.load_df(path , mapper = DFProcessor.load_mapper)
    return df

def load_df_max_date(path : Path | str , date_colname : str = 'date') -> int:
    """load dataframe from path"""
    path = Path(path)
    if not path.exists() or (df := load_df(path)).empty:
        return 19000101
    else:
        return int(max(df[date_colname]))

def load_df_min_date(path : Path | str , date_colname : str = 'date') -> int:
    """load dataframe from path"""
    path = Path(path)
    if not path.exists() or (df := load_df(path)).empty:
        return 99991231
    else:
        return int(min(df[date_colname]))

def load_dfs(
    paths : dict | list[Path] , * ,  
    key_column : str | None = 'date' , 
    parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , 
    mapper : Callable[[pd.DataFrame], pd.DataFrame] | None = None
) -> pd.DataFrame:
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
    if isinstance(paths , list):
        paths = {i:p for i,p in enumerate(paths)}
        key_column = None
    dfs = FileIOHandler.parallel_load_df(paths , parallel = parallel , mapper = mapper)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs , names = ['concat_df_index'])
    df = DFProcessor.load_mapper(df).reset_index(['concat_df_index'] , drop = False)
    if key_column is None:
        df = df.drop(columns = 'concat_df_index')
    else:
        df = df.drop(columns = [key_column], errors='ignore').rename(columns = {'concat_df_index':key_column})
    return df

def load_dfs_seperately(
    paths : dict | list[Path] , * ,  
    key_column : str | None = 'date' , 
    parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , 
    mapper : Callable[[pd.DataFrame], pd.DataFrame] | None = None
) -> dict[int | Any, pd.DataFrame]:
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
    if isinstance(paths , list):
        paths = {i:p for i,p in enumerate(paths)}
        key_column = None
    if mapper is None:
        def wrapped_mapper(df : pd.DataFrame) -> pd.DataFrame:
            return DFProcessor.load_mapper(df)
    else:
        def wrapped_mapper(df : pd.DataFrame) -> pd.DataFrame:
            return DFProcessor.load_mapper(mapper(df))

    dfs = FileIOHandler.parallel_load_df(paths , parallel = parallel , mapper = wrapped_mapper)
    if key_column is not None:
        dfs = {d:df.assign(**{key_column:d}) for d,df in dfs.items()}
    return dfs

def save_dfs_to_tar(dfs : dict[str , pd.DataFrame] , path : Path | str , *, overwrite = True , prefix = '' , indent = 1 , vb_level : Any = 1):
    """save multiple dataframes to tar file"""
    prefix = prefix or ''
    path = Path(path)
    path.parent.mkdir(parents=True , exist_ok=True)
    assert path.suffix == '.tar' , f'{path} is not a tar file'
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.unlink(missing_ok=True)
        FileIOHandler.save_tar(dfs , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def load_dfs_from_tar(path : str | Path , * , raise_if_not_exist = False) -> dict[str , pd.DataFrame]:
    """load multiple dataframes from tar file"""
    path = Path(path)
    if not path.exists():
        if raise_if_not_exist: 
            raise FileNotFoundError(path)
        else: 
            return {}
    dfs = FileIOHandler.load_tar(path , mapper = DFProcessor.load_mapper)
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

def min_date(db_src , db_key , *, use_alt = False):
    """get minimum date from any database data"""
    db_path = DBPath(db_src , db_key)
    return db_path.min_date(use_alt = use_alt)

def max_date(db_src , db_key , *, use_alt = False):
    """get maximum date from any database data"""
    db_path = DBPath(db_src , db_key)
    return db_path.max_date(use_alt = use_alt)

def save(df : pd.DataFrame | None , db_src : str , db_key : str , date = None , *, 
         overwrite = True , indent = 1 , vb_level : Any = 1 , reason : str = ''):
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
    df = DFProcessor.reset_index(df , reset = True)
    db_path = DBPath(db_src , db_key)
    mark = save_df(df , db_path.path_exact(date) , overwrite = overwrite , prefix = f'{db_src.title()} {reason}' if reason else db_key , 
                   indent = indent , vb_level = vb_level)
    return mark

def load(db_src , db_key , date = None , *, 
         date_colname = None , use_alt = False , closest = False , 
         raise_if_not_exist = False , indent = 1 , vb_level : Any = 1 , **kwargs) -> pd.DataFrame: 
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
    db_path = DBPath(db_src , db_key)
    df = load_df(db_path.path(date , use_alt = use_alt , closest = closest , indent = indent , vb_level = vb_level) , raise_if_not_exist = raise_if_not_exist)
    df = DFProcessor.load_process(df , date , date_colname , df_syntax = f'{db_path}' , indent = indent , vb_level = vb_level , **kwargs)
    return df

def loads(db_src , db_key , dates = None , start = None , end = None , *,
          date_colname = 'date' , use_alt = False , 
          parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , 
          fill_datavendor = False ,
          indent = 1 , vb_level : Any = 1 , **kwargs):
    """load multiple dates from database"""
    if DBPath.ByName(db_src):
        df = load(db_src , db_key , use_alt = use_alt , indent = indent , vb_level = vb_level , **kwargs)
    else:
        db_path = DBPath(db_src , db_key)
        if dates is None:
            assert start is not None or end is not None , f'start or end must be provided if dates is not provided'
            dates = db_path.dates(start , end , use_alt = use_alt)
        paths : dict[int , Path] = {int(date):db_path.path(date , use_alt = use_alt) for date in dates}
        df = load_dfs(paths , key_column = date_colname , parallel = parallel)
        df = DFProcessor.load_process(df , df_syntax = f'{db_src}/{db_key}/multi-dates' , indent = indent , vb_level = vb_level , **kwargs)
    if fill_datavendor:
        from src.data.loader import DATAVENDOR
        DATAVENDOR.db_loads_callback(df , db_src , db_key)
    return df

def rename(db_src , db_key , new_db_key):
    """rename database from db_key to new_db_key"""
    return DBPath(db_src , db_key).rename(new_db_key)

def path(db_src , db_key , date = None , * , use_alt = False) -> Path:
    """Get path of database
    Parameters
    ----------
    db_src: str
        database source name , or export source name (etc. pred , factor , market_factor , factor_stats_daily , factor_stats_weekly)
    db_key: str
        database key , or export key name
    date: int, default None
        date to be saved, if the db is by date, date is required
    """
    return DBPath(db_src , db_key).path(date , use_alt = use_alt)

def dates(db_src , db_key , * , start = None , end = None , year = None , use_alt = False):
    """get dates from any database data"""
    return DBPath(db_src , db_key).dates(start , end , year , use_alt = use_alt)
    
