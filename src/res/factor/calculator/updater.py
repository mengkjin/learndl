import numpy as np
import pandas as pd
import polars as pl

from abc import ABC , abstractmethod
from datetime import datetime
from typing import Any , Generator , Literal

from .factor_calc import FactorCalculator , WeightedPoolingCalculator

from src.proj import Logger , Proj , CALENDAR 
from src.proj.func import parallel    
from src.proj.util import SingletonMeta
from src.data import DATAVENDOR

__all__ = ['StockFactorUpdater' , 'MarketFactorUpdater' , 'AffiliateFactorUpdater' , 'PoolingFactorUpdater' , 'FactorStatsUpdater']

CATCH_ERRORS = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

class _BaseJob(ABC):
    """base job class"""
    def __init__(self , calc : FactorCalculator , overwrite : bool = False , vb_level : int = 1):
        self.calc = calc
        self.overwrite = overwrite
        self.vb_level = vb_level
        self.done = False

    def __repr__(self):
        return self.calc.factor_name

    def __call__(self , **kwargs) -> None:
        return self.go(**kwargs)
        
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
    def go(self , **kwargs) -> None:
        """do the job"""
        
    @abstractmethod
    def preview(self) -> None:
        """preview the job"""
        
class _JobFactorDate(_BaseJob):
    """single factor single date update job"""
    def __init__(self , calc : FactorCalculator , date : int , overwrite : bool = False , vb_level : int = 1 , **kwargs):
        super().__init__(calc , overwrite , vb_level)
        self.date = date

    def dates(self) -> np.ndarray:
        """dates of the job"""
        return np.array([self.date])

    def sort_key(self) -> Any: 
        """sort key of the job : (level , date , factor_name)"""
        return (self.level , self.date , self.factor_name)

    def go(self , indent : int = 1 , **kwargs) -> None:
        """do the job"""
        # self.preview(indent = indent , vb_level = vb_level) # will not preview since saving will output the information
        self.done = self.calc.update_day_factor(
            self.date , indent = indent , vb_level = self.vb_level , overwrite = self.overwrite , catch_errors = CATCH_ERRORS)
    
    def preview(self , indent : int = 1 , vb_level : int = 1) -> None:
        """preview the job"""
        Logger.stdout(f'{self.level} : {self.factor_name} at {self.date}' , indent = indent , vb_level = vb_level)

class _JobFactorAll(_BaseJob):
    """single factor update all job"""
    def __init__(self , calc : FactorCalculator , start : int | None = None , end : int | None = None , 
                 overwrite : bool = False , vb_level : int = 1 , **kwargs):
        super().__init__(calc , overwrite , vb_level)
        self.start = start
        self.end = end

    def dates(self) -> np.ndarray:
        """dates of the job"""
        return self.calc.target_dates(self.start , self.end , overwrite = self.overwrite)

    def sort_key(self) -> Any: 
        """sort key of the job"""
        return (self.level , self.factor_name)

    def go(self , indent : int = 1 , **kwargs) -> None:
        """do the job"""
        # self.preview(indent = indent , vb_level = self.vb_level) # will not preview since saving will output the information
        self.calc.update_all_factors(
            start = self.start , end = self.end , indent = indent , vb_level = self.vb_level , overwrite = self.overwrite
        )
        self.done = True

    def preview(self , indent : int = 1 , vb_level : int = 1) -> None:
        """preview the job"""
        Logger.stdout(f'Updating {self.level} : {self.factor_name} from {self.start} to {self.end}' , indent = indent , vb_level = vb_level)

class _JobFactorStats(_BaseJob):
    """single factor stats update all job"""
    def __init__(self , calc : FactorCalculator , stats_type : Literal['daily' , 'weekly'] , 
                 year : int , dates : np.ndarray , overwrite : bool = False , vb_level : int = 1 , **kwargs):
        super().__init__(calc , overwrite , vb_level)
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

    def go(self , indent : int = 1 , **kwargs) -> None:
        """do the job"""
        # self.preview(indent = indent , vb_level = self.vb_level) # will not preview since saving will output the information
        if self.stats_type == 'daily':
            self.calc.update_daily_stats(self.dates() , indent = indent , vb_level = self.vb_level , overwrite = self.overwrite)
        elif self.stats_type == 'weekly':
            self.calc.update_weekly_stats(self.dates() , indent = indent , vb_level = self.vb_level , overwrite = self.overwrite)
        else:
            raise ValueError(f'Invalid stats type: {self.stats_type}')
        self.done = True
    
    def preview(self , indent : int = 1 , vb_level : int = 1) -> None:
        """preview the job"""
        Logger.stdout(f'{self.level} : {self.factor_name} - {self.stats_type} at year {self.year}' , indent = indent , vb_level = vb_level)

class BaseFactorUpdater(metaclass=SingletonMeta):
    """manager of factor update jobs"""
    jobs : list[_BaseJob] | Any = None
    multi_thread : bool = False
    update_type : Literal['stock' , 'pooling' , 'affiliate' , 'market' , 'stats']
    
    def __repr__(self):
        n_jobs = len(self.jobs) if self.jobs is not None else 0
        return f'{self.__name__}({n_jobs} jobs)'

    @classmethod
    def jobs_dict(cls , jobs : list[_BaseJob]) -> dict[Any , _BaseJob]:
        """get jobs dictionary"""
        return {job:job for job in jobs}

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
        return list(FactorCalculator.iter(all , selected_factors , updatable = updatable , **kwargs))

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
                     overwrite = False , indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """
        collect FactorCalculator jobs for all factors between start and end date
        """
        cls.jobs.clear()
        
        if end is None: 
            end = min(CALENDAR.updated() , Proj.Conf.Factor.UPDATE.end)

        for calc in cls.calculators(all , selected_factors , **kwargs):
            if cls.update_type in ['affiliate']:
                ...
            elif cls.update_type in ['market' , 'pooling']:
                target_dates = calc.target_dates(start , end , overwrite = overwrite)
                if len(target_dates) > 0:
                    cls.jobs.append(_JobFactorAll(calc , start , end , overwrite , vb_level + 1))
            elif cls.update_type in ['stock']:
                target_dates = calc.target_dates(start , end , overwrite = overwrite)
                for date in calc.target_dates(start , end , overwrite = overwrite):
                    cls.jobs.append(_JobFactorDate(calc , date , overwrite , vb_level + 3))
            elif cls.update_type == 'stats':
                target_dates = calc.stats_target_dates(start , end , overwrite)
                for stats_type , dates in target_dates.items():
                    for year in np.unique(dates // 10000):
                        cls.jobs.append(_JobFactorStats(calc , stats_type , year , dates , overwrite , vb_level + 3))
            else:
                raise ValueError(f'Invalid update type: {cls.update_type}')
        cls.jobs.sort(key=lambda x: x.sort_key())

        if cls.jobs:
            Logger.success(f'Collecting {len(cls.jobs)} Jobs for {cls.__name__}' , indent = indent , vb_level = vb_level)
        else:
            Logger.skipping(f'There is no {cls.__name__} Jobs to Proceed...' , indent = indent , vb_level = vb_level)
        
    @classmethod
    def before_process_jobs(cls , start : int | None = None , end : int | None = None , 
                            all = True , selected_factors : list[str] | None = None ,
                            overwrite = False , indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """before process jobs"""
        for calc in cls.calculators(all , selected_factors , **kwargs):
            if isinstance(calc , WeightedPoolingCalculator):
                calc.drop_pooling_weight(after = start , overwrite = overwrite , indent = indent + 1 , vb_level = vb_level + 1)

    @classmethod
    def preview_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """preview update jobs for all factors between start and end date"""
        cls.collect_jobs(start , end , all , selected_factors , overwrite = overwrite , vb_level = 99)
        [job.preview() for job in cls.jobs]

    @classmethod
    def after_process_jobs(cls , indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """after process jobs"""
        failed_jobs = [job for job in cls.jobs if not job.done]
        if failed_jobs:
            if len(failed_jobs) <= 10:
                Logger.alert1(f'Remaining Failed Jobs: {failed_jobs}', indent = indent)
            else:
                Logger.alert1(f'Remaining {len(failed_jobs)} Jobs Failed: [{str(failed_jobs[:10])[:-1]},...]', indent = indent)
        elif len(cls.jobs) > 0:
            Logger.success(f'All {len(cls.jobs)} Jobs are Processed Successfully!' , indent = indent , vb_level = vb_level)

    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , indent : int = 0 , vb_level : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        default behavior is to collect jobs first to find all FactorCalculators then update one by one
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        """
        cls.collect_jobs(start , end , all , selected_factors , overwrite , indent = indent + 1 , vb_level = vb_level + 1)
        start_time = datetime.now()
        cls.before_process_jobs(start , end , all , selected_factors , overwrite , indent = indent + 1 , vb_level = vb_level + 1)
        for group , jobs in cls.grouped_jobs():
            cls.process_group_jobs(group , jobs , indent = indent + 2 , vb_level = vb_level + 2)

            if timeout > 0 and (datetime.now() - start_time).total_seconds() > timeout * 3600:
                Logger.alert1(f'Timeout reached, {timeout} hours passed, stopping update', indent = indent + 1)
                Logger.alert1(f'Terminated at {group}', indent = indent + 1)
                break
        cls.after_process_jobs(indent = indent + 1 , vb_level = vb_level + 1)
        

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
    def process_group_jobs(cls , group : dict , jobs : list[_BaseJob] , indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """process a group of jobs"""   
        raise NotImplementedError(f'process_group_jobs is not implemented for {cls.__name__}')

    @classmethod
    def update(cls , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        """update pooling factor data according"""
        Logger.note(f'Update : {cls.__name__} since last update!')
        cls.process_jobs(start = start , end = end , all = True , timeout = timeout)

    @classmethod
    def recalculate(cls , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        """recalculate factors between start and end date"""
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        Logger.note(f'Update : {cls.__name__} recalculate all factors!')
        cls.process_jobs(start = start , end = end , all = True , overwrite = True , timeout = timeout)

    @classmethod
    def rollback(cls , rollback_date : int , timeout : int = -1 , **kwargs) -> None:
        CALENDAR.check_rollback_date(rollback_date)
        start = int(CALENDAR.td(rollback_date , 1))
        Logger.note(f'Update : {cls.__name__} rollback from {rollback_date}!')
        cls.process_jobs(start = start , all = True , overwrite = True , timeout = timeout)
        
    @classmethod
    def fix(cls , factors : list[str] , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        assert factors , 'factors are required for fix'
        factors = [factor for factor in cls.factors() if factor in factors]
        if factors:
            Logger.note(f'Fixing : {cls.__name__} {factors} from {start} to {end}!')
            cls.process_jobs(selected_factors = factors , overwrite = True , start = start , end = end , timeout = timeout)

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
                           indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """process a group of stock factor update jobs"""
        DATAVENDOR.data_storage_control()
        Logger.stdout(f'Updating {group} : ' + (f'{len(jobs)} factors' if len(jobs) > 10 else str(jobs)) , indent = indent , vb_level = vb_level)
        parallel(cls.jobs_dict(jobs) , method = cls.multi_thread , indent = indent + 1)
        Logger.success(f'Stock Factor Update of {group} : {sum(job.done for job in jobs)} / {len(jobs)}' , indent = indent , vb_level = vb_level)
        if failed_jobs := [job for job in jobs if not job.done]: 
            Logger.alert1(f'Failed Stock Factors: {failed_jobs}', indent = indent)
            if cls.multi_thread:
                # if multi_thread is True, auto retry failed jobs
                Logger.stdout(f'Auto Retry Failed Stock Factors...' , indent = indent)
                parallel(cls.jobs_dict(failed_jobs) , method = cls.multi_thread , indent = indent + 1)
                if failed_jobs := [job for job in jobs if not job.done]:
                    Logger.alert1(f'Failed Stock Factors Again: {failed_jobs}', indent = indent)

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
    def process_group_jobs(cls , group : dict , jobs : list[_BaseJob] , 
                           indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """process a group of market factor update jobs"""
        assert len(jobs) == 1 , f'Only one job is allowed for group {group} , got {len(jobs)}'
        parallel(cls.jobs_dict(jobs) , method = cls.multi_thread , indent = indent + 1)

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
                           indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """process a group of market factor update jobs"""
        assert len(jobs) == 1 , f'Only one job is allowed for group {group} , got {len(jobs)}'
        parallel(cls.jobs_dict(jobs) , method = cls.multi_thread , indent = indent + 1)

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
                           indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """process a group of market factor update jobs"""
        assert len(jobs) == 1 , f'Only one job is allowed for group {group} , got {len(jobs)}'
        parallel(cls.jobs_dict(jobs) , method = cls.multi_thread , indent = indent + 1)
    
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
                           indent : int = 1 , vb_level : int = 2 , **kwargs) -> None:
        """process a group of factor stats update jobs"""
        Logger.stdout(f'Update Factor Stats of Year {group["year"]} : {len(jobs)} function calls , {sum([len(job.dates()) for job in jobs])} dates' , indent = indent , vb_level = vb_level)
        if len(jobs) <= 10:
            Logger.stdout(f'Jobs included: {jobs}' , indent = indent , vb_level = vb_level)
        parallel(cls.jobs_dict(jobs) , method = cls.multi_thread , indent = indent + 1)