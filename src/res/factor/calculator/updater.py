import numpy as np
import pandas as pd
import polars as pl
import time

from abc import ABC , abstractmethod
from typing import Any , Generator , Literal

from .factor_calc import FactorCalculator

from src.proj import Logger
from src.basic import CONF , CALENDAR 
from src.data import DATAVENDOR
from src.func.parallel import parallel
from src.func.singleton import SingletonMeta

__all__ = ['StockFactorUpdater' , 'MarketFactorUpdater' , 'AffiliateFactorUpdater' , 'PoolingFactorUpdater' , 'FactorStatsUpdater']

CATCH_ERRORS = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

class _BaseJob(ABC):
    """base job class"""
    def __init__(self , calc : FactorCalculator , verbose : bool = False , overwrite : bool = False):
        self.calc = calc
        self.verbose = verbose
        self.overwrite = overwrite
        self.done = False

    def __repr__(self):
        return self.calc.factor_name
        
    @property
    def level(self) -> str: 
        """level of the factor"""
        return self.calc.level

    @property
    def factor_name(self) -> str: 
        """name of the factor"""
        return self.calc.factor_name

    def match(self , **kwargs) -> bool:
        """check if the job matches the given attributes"""
        for key , value in kwargs.items():
            if getattr(self , key) != value:
                return False
        return True

    @abstractmethod
    def dates(self) -> np.ndarray:
        """dates of the job"""
        
    @abstractmethod
    def sort_key(self) -> Any: 
        """sort key of the job"""
        
    @abstractmethod
    def do(self) -> None:
        """do the job"""
        
    @abstractmethod
    def preview(self) -> None:
        """preview the job"""
        
class _JobFactorDate(_BaseJob):
    """single factor single date update job"""
    def __init__(self , calc : FactorCalculator , date : int , verbose : bool = False , overwrite : bool = False , **kwargs):
        super().__init__(calc , verbose , overwrite)
        self.date = date

    def dates(self) -> np.ndarray:
        """dates of the job"""
        return np.array([self.date])

    def sort_key(self) -> Any: 
        """sort key of the job : (level , date , factor_name)"""
        return (self.level , self.date , self.factor_name)

    def do(self) -> None:
        """do the job"""
        self.done = self.calc.update_day_factor(
            self.date , overwrite = self.overwrite , show_success = self.verbose , catch_errors = CATCH_ERRORS)
    
    def preview(self) -> None:
        """preview the job"""
        print(f'{self.level} : {self.factor_name} at {self.date}')

class _JobFactorAll(_BaseJob):
    """single factor update all job"""
    def __init__(self , calc : FactorCalculator , start : int | None = None , end : int | None = None , 
                 verbose : bool = False , overwrite : bool = False , **kwargs):
        super().__init__(calc , verbose , overwrite)
        self.start = start
        self.end = end

    def dates(self) -> np.ndarray:
        """dates of the job"""
        return self.calc.target_dates(self.start , self.end , overwrite = self.overwrite)

    def sort_key(self) -> Any: 
        """sort key of the job"""
        return (self.level , self.factor_name)

    def do(self) -> None:
        """do the job"""
        self.calc.update_all_factors(
            start = self.start , end = self.end , overwrite = self.overwrite , verbose = self.verbose
        )
        self.done = True

    def preview(self) -> None:
        """preview the job"""
        print(f'Updating {self.level} : {self.factor_name} from {self.start} to {self.end}')

class _JobFactorStats(_BaseJob):
    """single factor stats update all job"""
    def __init__(self , calc : FactorCalculator , stats_type : Literal['daily' , 'weekly'] , 
                 year : int , dates : np.ndarray , verbose : bool = False , overwrite : bool = False , **kwargs):
        super().__init__(calc , verbose , overwrite)
        self.stats_type = stats_type
        self.year = year
        self.year_dates = dates[dates // 10000 == year]

    def __repr__(self):
        return f'{self.calc.factor_name}({self.stats_type})'

    def dates(self) -> np.ndarray:
        """dates of the job"""
        return self.year_dates
        
    def sort_key(self) -> Any: 
        """sort key of the job"""
        return (self.year , self.factor_name)

    def do(self) -> None:
        """do the job"""
        if self.stats_type == 'daily':
            self.calc.update_daily_stats(self.dates() , overwrite = self.overwrite , verbose = self.verbose)
        elif self.stats_type == 'weekly':
            self.calc.update_weekly_stats(self.dates() , overwrite = self.overwrite , verbose = self.verbose)
        else:
            raise ValueError(f'Invalid stats type: {self.stats_type}')
        self.done = True
    
    def preview(self) -> None:
        """preview the job"""
        print(f'{self.level} : {self.factor_name} - {self.stats_type} at year {self.year}')

class BaseFactorUpdater(metaclass=SingletonMeta):
    """manager of factor update jobs"""
    jobs : list[_BaseJob] | Any = None
    multi_thread : bool = False
    update_type : Literal['stock' , 'pooling' , 'affiliate' , 'market' , 'stats']
    
    def __repr__(self):
        n_jobs = len(self.jobs) if self.jobs is not None else 0
        return f'{self.__name__}({n_jobs} jobs)'

    @classmethod
    def levels(cls) -> np.ndarray: 
        """unique levels of the jobs"""
        return np.unique([job.level for job in cls.jobs])

    @classmethod
    def sort_jobs(cls) -> None: 
        """sort jobs by sort_key"""
        cls.jobs.sort(key=lambda x: x.sort_key())

    @classmethod
    def calculators(cls , all = True , selected_factors : list[str] | None = None , updatable = True , **kwargs) -> list[FactorCalculator]:
        """get all calculators"""
        if cls.update_type == 'stats':
            kwargs = kwargs | {'is_market' : False}
        else:
            kwargs = kwargs | {'meta_type' : cls.update_type}
        return list(FactorCalculator.iter_calculators(all , selected_factors , updatable = updatable , **kwargs))

    @classmethod
    def factors(cls) -> list[str]:
        """get all factors"""
        return [calc.factor_name for calc in cls.calculators()]

    @staticmethod
    def filter_jobs(jobs : list , **kwargs) -> list[Any]:
        """filter jobs by level"""
        return [job for job in jobs if job.match(**kwargs)]

    @classmethod
    def collect_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     verbosity : int = 1 , overwrite = False , **kwargs) -> None:
        """
        collect FactorCalculator jobs for all factors between start and end date
        """
        cls.jobs.clear()
        
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)

        for calc in cls.calculators(all , selected_factors , **kwargs):
            if cls.update_type in ['affiliate']:
                ...
            elif cls.update_type in ['market' , 'pooling']:
                target_dates = calc.target_dates(start , end , overwrite = overwrite)
                if len(target_dates) > 0:
                    cls.jobs.append(_JobFactorAll(calc , start , end , verbosity - 1 > 0 , overwrite))
            elif cls.update_type in ['stock']:
                target_dates = calc.target_dates(start , end , overwrite = overwrite)
                for date in calc.target_dates(start , end , overwrite = overwrite):
                    cls.jobs.append(_JobFactorDate(calc , date , verbosity - 2 > 0 , overwrite))
            elif cls.update_type == 'stats':
                target_dates = calc.stats_target_dates(start , end , overwrite)
                for stats_type , dates in target_dates.items():
                    for year in np.unique(dates // 10000):
                        cls.jobs.append(_JobFactorStats(calc , stats_type , year , dates , verbosity - 2 > 0 , overwrite))
            else:
                raise ValueError(f'Invalid update type: {cls.update_type}')
        cls.jobs.sort(key=lambda x: x.sort_key())

        if cls.jobs:
            print(f'Finish Collecting {len(cls.jobs)} Factor Update - {cls.update_type.capitalize()} Jobs')
        else:
            print(f'There is no Factor Update - {cls.update_type.capitalize()} Jobs to Proceed...')
        
    @classmethod
    def preview_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """preview update jobs for all factors between start and end date"""
        cls.collect_jobs(start , end , all , selected_factors , overwrite = overwrite)
        [job.preview() for job in cls.jobs]

    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        default behavior is to collect jobs first to find all FactorCalculators then update one by one
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        """
        cls.collect_jobs(start , end , all , selected_factors , verbosity , overwrite)
        start_time = time.time()
        for group , jobs in cls.grouped_jobs():
            cls.process_group_jobs(group , jobs , verbosity)

            if timeout > 0 and (time.time() - start_time) > timeout * 3600:
                Logger.fail(f'Timeout: {timeout} hours reached, stopping update')
                Logger.fail(f'Terminated at {group}')
                break

    @classmethod
    def grouped_jobs_by(cls , keys : list[str]) -> Generator[tuple[dict , list[Any]] , None , None]:
        """group jobs by key"""
        group_values = sorted(set(tuple(getattr(job , key) for key in keys) for job in cls.jobs))
        for gv in group_values:
            group = dict(zip(keys , gv))
            yield group , cls.filter_jobs(cls.jobs , **group)

    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[dict , list[_BaseJob]] , None , None]:
        """
        group jobs by level and date
        eg. level = 'level1' , factor_name = 'factor1'
        groups = sorted(set((job.level , job.factor_name) for job in cls.jobs))
        for level , factor_name in groups:
            yield (level , factor_name) , cls.filter_jobs(cls.jobs , level = level , factor_name = factor_name)
        """
        raise NotImplementedError(f'grouped_jobs is not implemented for {cls.__name__}')
        
    @classmethod
    def process_group_jobs(cls , group : dict , jobs : list[_BaseJob] , verbosity : int = 1 , **kwargs) -> None:
        """process a group of jobs"""   
        raise NotImplementedError(f'process_group_jobs is not implemented for {cls.__name__}')

    @classmethod
    def update(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        """update pooling factor data according"""
        cls.process_jobs(start = start , end = end , all = True , verbosity = verbosity , timeout = timeout)

    @classmethod
    def recalculate(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        """update pooling factor data according"""
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        cls.process_jobs(start = start , end = end , all = True , overwrite = True , verbosity = verbosity , timeout = timeout)

    @classmethod
    def rollback(cls , rollback_date : int , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        CALENDAR.check_rollback_date(rollback_date)
        start = CALENDAR.td(rollback_date , 1)
        cls.process_jobs(start = start , all = True , overwrite = True , verbosity = verbosity , timeout = timeout)
        
    @classmethod
    def fix(cls , factors : list[str] , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        assert factors , 'factors are required for fix'
        print(f'Fixing {(cls.update_type.capitalize() + " ") if cls.update_type else ''}Factor Calculations: {factors}')
        cls.process_jobs(selected_factors = factors , overwrite = True , start = start , end = end , verbosity = verbosity , timeout = timeout)

    @classmethod
    def eval_coverage(cls , selected_factors : list[str] | None = None , **kwargs) -> pd.DataFrame:
        """
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        """
        dfs : list[pd.DataFrame] = []

        def load_fac(calc : FactorCalculator , date : int):
            factor = calc.eval_factor(date)
            valid_count = factor.loc[:,calc.factor_name].notna().sum()
            return pd.DataFrame({'factor' : [calc.factor_name] , 'date' : [date] , 'valid_count' : [valid_count]})
            
        for calc in cls.calculators(selected_factors = selected_factors , **kwargs):
            dates = calc.stored_dates()
            calls = {date:(load_fac , {'calc' : calc , 'date' : date}) for date in dates}
            factor_coverage = parallel(calls , method = cls.multi_thread)
            dfs.extend(list(factor_coverage.values()))

        df = pd.concat(dfs)
        grouped = df.groupby(by='factor')
        stats : dict[str , pd.Series | Any] = {}
        stats['mean'] = grouped.mean()['valid_count']
        stats['min'] = grouped.min()['valid_count']
        stats['max'] = grouped.max()['valid_count']
        stats['std'] = grouped.std()['valid_count']
        agg = pd.concat([stats[key].rename(key) for key in stats.keys()], axis = 1)
        df.to_excel('factor_coverage.xlsx' , sheet_name='full coverage')
        agg.to_excel('factor_coverage_agg.xlsx' , sheet_name='coverage_agg_stats')
        return df

class StockFactorUpdater(BaseFactorUpdater):
    """manager of factor update jobs"""
    jobs : list[_BaseJob] = []
    multi_thread : bool = True
    update_type = 'stock'

    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[Any , list[_BaseJob]] , None , None]:
        """group jobs by level and date"""
        return cls.grouped_jobs_by(['level' , 'date'])

    @classmethod
    def process_group_jobs(cls , group : dict , jobs : list[_BaseJob] , 
                           verbosity : int = 1 , **kwargs) -> None:
        """process a group of stock factor update jobs"""
        DATAVENDOR.data_storage_control()
        if verbosity > 1:
            print(f'Updating {group} : ' + (f'{len(jobs)} factors' if len(jobs) > 10 else str(jobs)))
        parallel({job:job.do for job in jobs} , method = cls.multi_thread)
        if verbosity > 0:
            print(f'Stock Factor Update of {group} Done: {sum(job.done for job in jobs)} / {len(jobs)}')
        if failed_jobs := [job for job in jobs if not job.done]: 
            print(f'Failed Stock Factors: {failed_jobs}')
            if cls.multi_thread:
                # if multi_thread is True, auto retry failed jobs
                print(f'Auto Retry Failed Stock Factors...')
                parallel({job:job.do for job in failed_jobs} , method = cls.multi_thread)
                if failed_jobs := [job for job in jobs if not job.done]:
                    print(f'Failed Stock Factors Again: {failed_jobs}')

class MarketFactorUpdater(BaseFactorUpdater):
    """manager of market factor update jobs"""
    jobs : list[_JobFactorAll] = []
    multi_thread : bool = False
    update_type = 'market'

    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[dict , list[_JobFactorAll]] , None , None]:
        """group jobs by level and factor_name"""
        return cls.grouped_jobs_by(['level' , 'factor_name'])
        
    @classmethod
    def process_group_jobs(cls , group : dict , jobs : list[_JobFactorAll] , 
                           verbosity : int = 1 , **kwargs) -> None:
        """process a group of market factor update jobs"""
        assert len(jobs) == 1 , f'Only one job is allowed for group {group} , got {len(jobs)}'
        if verbosity > 0:
            jobs[0].preview()
        jobs[0].do()

class PoolingFactorUpdater(BaseFactorUpdater):
    """manager of pooling factor update jobs"""
    jobs : list[_BaseJob] = []
    multi_thread : bool = False
    update_type = 'pooling'

    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[dict , list[_BaseJob]] , None , None]:
        """group jobs by level and factor_name"""
        return cls.grouped_jobs_by(['level' , 'factor_name'])
        
    @classmethod
    def process_group_jobs(cls , group : dict , jobs : list[_BaseJob] , 
                           verbosity : int = 1 , **kwargs) -> None:
        """process a group of market factor update jobs"""
        assert len(jobs) == 1 , f'Only one job is allowed for group {group} , got {len(jobs)}'
        if verbosity > 0:
            jobs[0].preview()
        jobs[0].do()

class AffiliateFactorUpdater(BaseFactorUpdater):
    """manager of affiliate factor update jobs"""
    jobs : list[_BaseJob] = []
    multi_thread : bool = False
    update_type = 'affiliate'

    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[Any , list[_BaseJob]] , None , None]:
        """group jobs by level and factor_name"""
        return cls.grouped_jobs_by(['level' , 'factor_name'])

    @classmethod
    def process_group_jobs(cls , group : Any , jobs : list[_BaseJob] , 
                           verbosity : int = 1 , **kwargs) -> None:
        """process a group of market factor update jobs"""
        assert len(jobs) == 1 , f'Only one job is allowed for group {group} , got {len(jobs)}'
        if verbosity > 0:
            jobs[0].preview()
        jobs[0].do()
    
class FactorStatsUpdater(BaseFactorUpdater):
    """manager of factor stats update jobs"""
    jobs : list[_BaseJob] = []
    multi_thread : bool = False
    update_type = 'stats'

    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[dict , list[_BaseJob]] , None , None]:
        """group jobs by level and date"""
        return cls.grouped_jobs_by(['year'])

    @classmethod
    def process_group_jobs(cls , group : Any , jobs : list[_BaseJob] , 
                           verbosity : int = 1 , **kwargs) -> None:
        """process a group of factor stats update jobs"""
        print(f'Update Factor Stats of Year {group["year"]} : {len(jobs)} function calls , {sum([len(job.dates()) for job in jobs])} dates')
        if verbosity > 1 and len(jobs) <= 10:
            print(f'Jobs included: {jobs}')
        parallel({job:job.do for job in jobs} , method = cls.multi_thread)