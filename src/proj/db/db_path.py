"""Structure of database path."""
from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import Any , Generator , Mapping , Iterable

from src.proj.env import PATH
from src.proj.log import Logger
from src.proj.core import strPath , strPaths

from .core import DATAFRAME_SUFFIX , SRC_ALTERNATIVES , DB_BY_NAME , DB_BY_DATE , EXPORT_BY_NAME , EXPORT_BY_DATE

__all__ = ['DBPath' , 'path_date' , 'file_dates' , 'dir_dates' , 'rename' , 'path' , 'dates' , 'paths' , 'min_date' , 'max_date']

def path_date(path : strPath) -> int:
    """get date from path"""
    return int(Path(path).stem[-8:])

def file_dates(paths : strPaths , startswith = '' , endswith = '') -> np.ndarray[int, Any]:
    """get dates from paths"""
    if isinstance(paths , Mapping):
        paths = paths.values()
    paths = [Path(p) for p in paths]
    datestrs = [p.stem[-8:] for p in paths if p.name.startswith(startswith) and p.name.endswith(endswith)]
    dates = np.array([ds for ds in datestrs if ds.isdigit() and len(ds) == 8] , dtype = int)
    dates.sort()
    return dates

def dir_dates(directory : Path , start = None , end = None , year = None):
    """get dates from directory"""
    paths = directory.rglob('*')
    dates = file_dates(paths)
    if end   is not None: 
        dates = dates[dates <= end]
    if start is not None: 
        dates = dates[dates >= start]
    if year is not None:
        dates = dates[dates // 10000 == year]
    return dates
    
class DBPath:
    """DB Path structure of db_src and db_key"""
    src_alternatives = SRC_ALTERNATIVES
    db_by_name = DB_BY_NAME
    db_by_date = DB_BY_DATE
    export_by_name = EXPORT_BY_NAME
    export_by_date = EXPORT_BY_DATE

    instance_cache : dict[str , DBPath] = {}

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

    def syntax(self , date : int | list[int] | np.ndarray[int, Any] | None = None) -> str:
        """get syntax of database"""
        if self.by_name:
            return f'{self.src}.{self.key}'
        else:
            if date is None:
                date = []
            elif not isinstance(date , Iterable):
                date = [date]
            if len(date) == 0:
                date_str = 'nodate'
            elif len(date) == 1:
                date_str = str(date[0])
            else:
                date_str = f'{min(date)}-{max(date)}'
            return f'{self.src}.{self.key}.{date_str}'

    def years(self) -> list[int]:
        """get years from database"""
        directory = self.parent
        return [int(y.stem) for y in directory.iterdir() if y.is_dir() and any(y.iterdir())] if directory.exists() else []

    def dates(self , start = None , end = None , year = None , * , use_alt = False) -> np.ndarray[int, Any]:
        """get dates from any database data"""
        if use_alt:
            candidates = [self] + self.alternatives()
            dates = np.unique(np.concatenate([db_path.dates(start , end , year , use_alt = False) for db_path in candidates]))
        else:
            directory = self.parent.joinpath(str(year)) if year else self.parent
            dates = dir_dates(directory , start , end)
        return np.array(dates, dtype=int)
    
    def min_date(self , * , use_alt = False) -> int:
        """get minimum date from any database data"""
        if use_alt:
            candidates = [self] + self.alternatives()
            return min(db_path.min_date(use_alt = False) for db_path in candidates)
        else:
            directory = self.parent
            years = self.years()
            if years: 
                dates = file_dates(directory.joinpath(str(min(years))).iterdir())
                mdate = min(dates) if len(dates) else 99991231
            else:
                mdate = 99991231
            return int(mdate)

    def max_date(self , * , use_alt = False) -> int:
        """get maximum date from any database data"""
        if use_alt:
            candidates = [self] + self.alternatives()
            return max(db_path.max_date(use_alt = False) for db_path in candidates)
        else:
            directory = self.parent
            years = self.years()
            if years: 
                dates = file_dates(directory.joinpath(str(max(years))).iterdir())
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

    def get_paths(self , dates : np.ndarray | list[int] | None = None , start : int | None = None , end : int | None = None , year = None , 
                  use_alt = False , closest = False) -> dict[int, Path]:
        """get paths from database"""
        if dates is None:
            assert start is not None or end is not None , f'start or end must be provided if dates is not provided'
            dates = self.dates(start , end , use_alt = use_alt)
        paths = {date:self.path(date , use_alt = use_alt) for date in dates}
        paths = {date:path for date,path in paths.items() if path.exists()}
        if closest and len(dates) > 0 and min(dates) not in paths:
            prev_date = self.date_closest(min(dates))
            if prev_date is not None:
                paths[prev_date] = self.path(prev_date , use_alt = use_alt)
        return paths

    def alternatives(self) -> list[DBPath]:
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


def rename(db_src : str , db_key : str , new_db_key : str) -> None:
    """rename database from db_key to new_db_key"""
    return DBPath(db_src , db_key).rename(new_db_key)

def path(db_src : str , db_key : str , date : int | None = None , * , use_alt = False) -> Path:
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

def dates(db_src : str , db_key : str , start : int | None = None , end : int | None = None , year = None , use_alt = False) -> np.ndarray[int, Any]:
    """get dates from any database data"""
    db_path = DBPath(db_src , db_key)
    return db_path.dates(start , end , year , use_alt = use_alt)

def paths(db_src : str , db_key : str , * , dates : np.ndarray | list[int] | None = None , start : int | None = None , end : int | None = None , year = None , use_alt = False) -> list[Path]:
    """get paths from any database data"""
    db_path = DBPath(db_src , db_key)
    if db_path.by_name:
        paths = [db_path.path_exact()]
    else:
        if dates is None:
            assert start is not None or end is not None , f'start or end must be provided if dates is not provided'
            dates = db_path.dates(start , end , use_alt = use_alt)
        else:
            dates = np.array(dates , dtype = int)
        paths = [db_path.path(date , use_alt = use_alt) for date in dates]
        paths = [path for path in paths if path.exists()]
    return paths
    
def min_date(db_src : str , db_key : str , *, use_alt = False) -> int:
    """get minimum date from any database data"""
    db_path = DBPath(db_src , db_key)
    return db_path.min_date(use_alt = use_alt)

def max_date(db_src : str , db_key : str , *, use_alt = False) -> int:
    """get maximum date from any database data"""
    db_path = DBPath(db_src , db_key)
    return db_path.max_date(use_alt = use_alt)