import numpy as np
import pandas as pd
import polars as pl
import time

from dataclasses import dataclass
from typing import Any , Generator , Callable

from .factor_calc import FactorCalculator

from src.proj import Logger
from src.basic import CONF , CALENDAR 
from src.data import DATAVENDOR
from src.func.parallel import parallels
from src.func.singleton import singleton

__all__ = ['StockFactorUpdater' , 'PoolingFactorUpdater' , 'FactorStatsUpdater']

CATCH_ERRORS = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

def iter_calcs(all = True , selected_factors : list[str] | None = None ,
               is_pooling : bool | None = None , updatable : bool | None = True , **kwargs) -> Generator[FactorCalculator , None , None]:
    """iterate over calculators"""
    return FactorCalculator.iter_calculators(all , selected_factors , updatable = updatable , is_pooling = is_pooling , **kwargs)

def eval_coverage(selected_factors : list[str] | None = None , **kwargs) -> pd.DataFrame:
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

    for calc in iter_calcs(selected_factors = selected_factors , **kwargs):
        dates = calc.stored_dates()
        def load_fac(date : int):
            factor = calc.eval_factor(date)
            valid_count = factor.loc[:,calc.factor_name].notna().sum()
            return pd.DataFrame({'factor' : [calc.factor_name] , 'date' : [date] , 'valid_count' : [valid_count]})
        calls = [(load_fac , (date , ) , {}) for date in dates]
        factor_coverage = parallels(calls , method = 'threading')
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

@dataclass  
class FactorUpdateJob:
    """single factor update job"""
    calc : FactorCalculator
    date : int

    def __post_init__(self):
        self.done = False
        
    @property
    def level(self) -> str: 
        """level of the factor"""
        return self.calc.level
    @property
    def factor_name(self) -> str: 
        """name of the factor"""
        return self.calc.factor_name
    @property
    def sort_key(self) -> Any: 
        """sort key of the job : (level , date , factor_name)"""
        return (self.level , self.date , self.factor_name)
    def do(self , show_success : bool = False , overwrite = False) -> None:
        """do the job"""
        self.done = self.calc.update_day_factor(
            self.date , overwrite = overwrite , show_success = show_success , catch_errors = CATCH_ERRORS)
    
@singleton
class StockFactorUpdater:
    """manager of factor update jobs"""
    jobs : list[FactorUpdateJob] = []
    multi_thread : bool = True
    
    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.jobs)} jobs)'

    @classmethod
    def calculators(cls) -> list[FactorCalculator]:
        """get all calculators"""
        return list(iter_calcs(is_pooling = False))

    @classmethod
    def factors(cls) -> list[str]:
        """get all factors"""
        return [calc.factor_name for calc in cls.calculators()]

    @classmethod
    def levels(cls) -> np.ndarray: 
        """unique levels of the jobs"""
        return np.unique([job.level for job in cls.jobs])
    @classmethod
    def dates(cls) -> np.ndarray:  
        """unique dates of the jobs"""
        return np.unique([job.date for job in cls.jobs])
    @classmethod
    def groups(cls) -> list[tuple[str , int]]: 
        """groups of update jobs (by level and date)"""
        return sorted(set((job.level , job.date) for job in cls.jobs))
    @classmethod
    def to_dataframe(cls) -> pd.DataFrame:
        """convert to dataframe for illustration"""
        columns = ['level' , 'date' , 'factor']
        return pd.DataFrame(
            [(job.level , job.date , job.factor_name) for job in cls.jobs] , 
            columns=pd.Index(columns)
        ).sort_values(by=columns)

    @classmethod
    def clear(cls) -> None: 
        """clear all jobs"""
        cls.jobs.clear()
    @classmethod
    def sort(cls) -> None: 
        """sort jobs by sort_key"""
        cls.jobs.sort(key=lambda x: x.sort_key)
    @classmethod
    def append(cls , job : FactorUpdateJob) -> None: 
        """append a job"""
        cls.jobs.append(job)
    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[tuple[str , int] , list[FactorUpdateJob]] , None , None]:
        """group jobs by level and date"""
        for level , date in cls.groups():
            yield (level , date) , cls.filter_jobs(cls.jobs , level , date)
    @staticmethod
    def filter_jobs(jobs : list[FactorUpdateJob] , level : str , date : int) -> list[FactorUpdateJob]:
        """filter jobs by level and date"""
        return [job for job in jobs if job.level == level and job.date == date]

    @classmethod
    def unfinished_factors(cls , date : int | None = None) -> dict[int , list[FactorCalculator]]:
        """get unfinished factors"""
        factors : dict[int , list[FactorCalculator]] = {}
        for calc in iter_calcs(updatable = True , is_pooling = False):
            if date is None:
                for d in calc.target_dates():
                    if d not in factors:
                        factors[d] = []
                    factors[d].append(calc)
            else:
                if date in calc.target_dates():
                    if date not in factors:
                        factors[date] = []
                    factors[date].append(calc)
        return factors

    @classmethod
    def collect_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        """
        selected_factors = selected_factors or []
        cls.clear()
        
        if not (all or selected_factors or kwargs): 
            return
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)

        for calc in iter_calcs(all , selected_factors , is_pooling = False , **kwargs):
            for date in calc.target_dates(start , end , overwrite = overwrite):
                cls.append(FactorUpdateJob(calc , date))

        if len(cls.jobs) == 0:
            print('There is no Factor Update Jobs to Proceed...')
        else:
            levels , dates = cls.levels() , cls.dates()
            print(f'Finish Collecting {len(cls.jobs)} Factor Update Jobs , levels: {levels} , dates: {min(dates)} ~ {max(dates)}')
    
    @classmethod
    def clear_jobs(cls , date : int , verbosity : int = 1) -> None:
        """clear jobs of a given date"""
        if verbosity > 0:
            print(f'Clearing factors of {date}')

        removed_factors = []
        for calc in iter_calcs(is_pooling = False):
            cleared = calc.clear_stored_data(date)
            if cleared:
                removed_factors.append(calc.factor_name)
        if verbosity > 0:
            print(f'Removed {len(removed_factors)} factors')

    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None ,
                     all = True , selected_factors : list[str] | None = None ,
                     verbosity : int = 1 , overwrite = False , timeout : int = -1 , auto_retry = True) -> None:
        """
        perform all update jobs

        verbosity : 
            0 : show only error
            1 : show error and success stats
            2 : show all
        overwrite : if True , overwrite existing data
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        """
        cls.collect_jobs(start , end , all , selected_factors , overwrite)

        if len(cls.jobs) == 0: 
            return

        def do_job(job : FactorUpdateJob): 
            job.do(verbosity > 2 , overwrite)

        start_time = time.time()
        timeout = timeout * 3600
        for (level , date) , jobs in cls.grouped_jobs():
            DATAVENDOR.data_storage_control()
            keys = [job.factor_name for job in jobs]
            if verbosity > 1:
                if len(keys) > 10:
                    print(f'Updating {level} at {date} : {len(keys)} factors')
                else:
                    print(f'Updating {level} at {date} : {keys}')
            calls = [(do_job , (job , ) , {}) for job in jobs]
            parallels(calls , keys = keys , method = cls.multi_thread)
            failed_jobs = [job for job in jobs if not job.done]
            if verbosity > 0:
                print(f'Factor Update of {level} at {date} Done: {len(jobs) - len(failed_jobs)} / {len(jobs)}')
                if failed_jobs: 
                    print(f'Failed Factors: {[job.factor_name for job in failed_jobs]}')
            if auto_retry and failed_jobs:
                if verbosity > 0: 
                    print(f'Auto Retry Failed Factors...')
                calls = [(do_job , (job , ) , {}) for job in failed_jobs]
                parallels(calls , method = len(failed_jobs) > 10)
                failed_again_jobs = [job for job in failed_jobs if not job.done]
                if failed_again_jobs:
                    print(f'Failed Factors Again: {[job.factor_name for job in failed_again_jobs]}')
            if timeout > 0 and (time.time() - start_time) > timeout:
                Logger.warning(f'Timeout: {timeout} hours reached, stopping update')
                Logger.warning(f'Terminated at level {level} at date {date}')
                break
        [cls.jobs.remove(job) for job in jobs if job.done]

    @classmethod
    def update(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1) -> None:
        """update factor data according"""
        cls.process_jobs(start , end , all = True , verbosity = verbosity , timeout = timeout)

    @classmethod
    def recalculate(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1) -> None:
        """update factor data according"""
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        cls.process_jobs(start , end , verbosity = verbosity , overwrite = True , timeout = timeout)
        
    @classmethod
    def rollback(cls , rollback_date : int , verbosity : int = 1 , timeout : int = -1) -> None:
        CALENDAR.check_rollback_date(rollback_date)
        start = CALENDAR.td(rollback_date , 1)
        cls.process_jobs(start , verbosity = verbosity , overwrite = True , timeout = timeout)
        
    @classmethod
    def fix(cls , factors : list[str] , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1) -> None:
        assert factors , 'factors are required for fix'
        print(f'Fixing Factor Calculations: {factors}')
        cls.process_jobs(start , end , selected_factors = factors , verbosity = verbosity , overwrite = True , timeout = timeout)

@singleton
class PoolingFactorUpdater:
    """manager of factor update jobs"""
    multi_thread : bool = False
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @classmethod
    def calculators(cls) -> list[FactorCalculator]:
        """get all calculators"""
        return list(iter_calcs(is_pooling = True))

    @classmethod
    def factors(cls) -> list[str]:
        """get all factors"""
        return [calc.factor_name for calc in cls.calculators()]

    @classmethod
    def grouped_calculators(cls , all = True , selected_factors : list[str] | None = None , **kwargs) -> list[tuple[str , list[FactorCalculator]]]:
        """group jobs by level and date"""
        calculators = list(iter_calcs(all , selected_factors , is_pooling = True , **kwargs))
        groups : dict[str , list[FactorCalculator]] = {}
        for calc in calculators:
            if calc.level not in groups:
                groups[calc.level] = []
            groups[calc.level].append(calc)
        return sorted(groups.items() , key = lambda x: x[0])

    @classmethod
    def preview_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """preview update jobs for all factors between start and end date"""
        selected_factors = selected_factors or []
        
        if not (all or selected_factors or kwargs): 
            return
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)

        for level , calculators in cls.grouped_calculators(all , selected_factors , **kwargs):
            for calc in calculators:
                print(f'{level} : {calc.factor_name} at {start} ~ {end}')
        
    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        """
        selected_factors = selected_factors or []
        
        if not (all or selected_factors or kwargs): 
            return
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)
        
        grouped_calculators = cls.grouped_calculators(all , selected_factors , **kwargs)
        if len(grouped_calculators) == 0:
            print('There is no Pooling Factor Update Jobs to Proceed...')
        else:
            n_jobs = len(grouped_calculators)
            levels = sorted(list(set([level for level , _ in grouped_calculators])))
            print(f'Finish Collecting {n_jobs} Pooling Factor Update Jobs , levels: {levels} , number of factors: {len(grouped_calculators)}')

        start_time = time.time()
        timeout = timeout * 3600
        for level , calculators in grouped_calculators:
            for calc in calculators:
                if verbosity > 0:
                    print(f'Updating {level} : {calc.factor_name} at {start} ~ {end}')
                calc.update_all_factors(start = start , end = end , overwrite = overwrite , verbose = verbosity > 1)

                if timeout > 0 and (time.time() - start_time) > timeout:
                    Logger.warning(f'Timeout: {timeout} hours reached, stopping update')
                    Logger.warning(f'Terminated at level {level} , factor {calc.factor_name}')
                    break
                
    
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
        print(f'Fixing Pooling Factor Calculations: {factors}')
        cls.process_jobs(selected_factors = factors , overwrite = True , start = start , end = end , verbosity = verbosity , timeout = timeout)

@singleton
class FactorStatsUpdater:
    """manager of factor stats update jobs"""
    multi_thread : bool = False
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @classmethod
    def calculators(cls) -> list[FactorCalculator]:
        """get all calculators"""
        return list(iter_calcs())

    @classmethod
    def factors(cls) -> list[str]:
        """get all factors"""
        return [calc.factor_name for calc in cls.calculators()]
    
    @classmethod
    def update_factor_stats(cls , start : int | None = None , end : int | None = None , overwrite = False , 
                            all = True , selected_factors : list[str] | None = None , **kwargs):
        """update all factor stats"""
        func_calls : dict[int , list[tuple[Callable , tuple[Any,...] , dict[str , Any] | None]]] = {}
        for calc in iter_calcs(all , selected_factors , **kwargs):
            target_dates = calc.stats_target_dates(start , end , overwrite)
            for stats_type , dates in target_dates.items():
                func  = getattr(calc , f'update_{stats_type}_stats')
                years = np.unique(dates // 10000)
                for year in years:
                    if year not in func_calls:
                        func_calls[year] = []
                    func_calls[year].append((func , (dates[dates // 10000 == year] , ) , {'overwrite' : overwrite}))
        
        func_calls = {year: calls for year , calls in sorted(func_calls.items())}
        if len(func_calls) == 0:
            print('There is no Factor Stats Update to Proceed')
            return

        total_calls , total_dates = 0 , 0
        for year , calls in func_calls.items():
            n_calls = len(calls)
            n_dates = sum([len(args[0]) for _ , args , _ in calls])
            print(f'Update Factor Stats of Year {year} : {n_calls} function calls , {n_dates} dates')
            parallels(calls , method = 'forloop')
            total_calls += n_calls
            total_dates += n_dates

        print(f'Factor Stats Update Done: {len(func_calls)} Years , {total_calls} function calls , {total_dates} dates')

    @classmethod
    def update(cls , start : int | None = None , end : int | None = None , **kwargs) -> None:
        """update factor stats"""
        cls.update_factor_stats(start , end)

    @classmethod
    def recalculate(cls , start : int | None = None , end : int | None = None , **kwargs) -> None:
        """recalculate factor stats"""
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        cls.update_factor_stats(start , end , overwrite = True)

    @classmethod
    def rollback(cls , rollback_date : int , **kwargs) -> None:
        """update factor stats rollback from a given date"""
        CALENDAR.check_rollback_date(rollback_date)
        start = CALENDAR.td(rollback_date , 1)
        cls.update_factor_stats(start , overwrite = True)
        
    @classmethod
    def fix(cls , factors : list[str] , start : int | None = None , end : int | None = None , **kwargs) -> None:
        assert factors , 'factors are required for fix'
        print(f'Fixing Factor Stats: {factors}')
        cls.update_factor_stats(start , end , overwrite = True , selected_factors = factors)
