import numpy as np
import pandas as pd
import polars as pl

from dataclasses import dataclass
from typing import Any , Generator , Callable

from .factor_calc import FactorCalculator

from src.basic import CONF , CALENDAR
from src.data import DATAVENDOR
from src.func.parallel import parallel , parallels

CATCH_ERRORS = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

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
        self.done = self.calc.update_day_factor(self.date , overwrite = overwrite , show_success = show_success , catch_errors = CATCH_ERRORS)
    
class FactorUpdateJobManager:
    """manager of factor update jobs"""
    _instance = None
    jobs : list[FactorUpdateJob] = []
    multi_thread : bool = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return f'FactorUpdateJobs({len(self.jobs)} jobs)'

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
        for calc in FactorCalculator.iter_calculators():
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
                     all_factors = False , selected_factors : list[str] | None = None ,
                     overwrite = False , groups_in_one_update : int | None = None , 
                     **kwargs) -> None:
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
        
        if not (all_factors or selected_factors or kwargs): 
            return
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)

        for calc in cls.iter_calculators(all_factors , selected_factors , **kwargs):
            for date in calc.target_dates(start , end , overwrite = overwrite):
                cls.append(FactorUpdateJob(calc , date))

        if len(cls.jobs) == 0:
            print('There is no Factor Update Jobs to Proceed...')
        else:
            if groups_in_one_update is not None:
                groups = cls.groups()[:groups_in_one_update]
                cls.jobs = [job for level , date in groups for job in cls.filter_jobs(cls.jobs , level , date)]
            levels , dates = cls.levels() , cls.dates()
            print(f'Finish Collecting {len(cls.jobs)} Factor Update Jobs , levels: {levels} , dates: {min(dates)} ~ {max(dates)}')
    
    @classmethod
    def process_jobs(cls , verbosity : int = 1 , overwrite = False , auto_retry = True) -> None:
        """
        perform all update jobs

        verbosity : 
            0 : show only error
            1 : show error and success stats
            2 : show all
        overwrite : if True , overwrite existing data
        """
        if len(cls.jobs) == 0: 
            return

        def do_job(job : FactorUpdateJob): 
            job.do(verbosity > 1 , overwrite)

        for (level , date) , jobs in cls.grouped_jobs():
            DATAVENDOR.data_storage_control()
            parallel(do_job , jobs , keys = [job.factor_name for job in jobs] , method = cls.multi_thread)
            failed_jobs = [job for job in jobs if not job.done]
            if verbosity > 0:
                print(f'Factor Update of {level} at {date} Done: {len(jobs) - len(failed_jobs)} / {len(jobs)}')
                if failed_jobs: 
                    print(f'Failed Factors: {[job.factor_name for job in failed_jobs]}')
            if auto_retry and failed_jobs:
                if verbosity > 0: 
                    print(f'Auto Retry Failed Factors...')
                parallel(do_job , failed_jobs , method = len(failed_jobs) > 10)
                failed_again_jobs = [job for job in failed_jobs if not job.done]
                if failed_again_jobs:
                    print(f'Failed Factors Again: {[job.factor_name for job in failed_again_jobs]}')
        [cls.jobs.remove(job) for job in jobs if job.done]

    @classmethod
    def update_factor_stats(cls , start : int | None = None , end : int | None = None , overwrite = False , all_factors = False , selected_factors : list[str] | None = None , **kwargs):
        """update all factor stats"""
        func_calls : dict[int , list[tuple[Callable , tuple[Any,...] , dict[str , Any] | None]]] = {}
        for calc in cls.iter_calculators(all = all_factors , selected_factors = selected_factors , **kwargs):
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
    def clear_jobs(cls , date : int , verbosity : int = 1) -> None:
        """clear jobs of a given date"""
        if verbosity > 0:
            print(f'Clearing factors of {date}')

        removed_factors = []
        for calc in cls.iter_calculators(all_factors = True):
            cleared = calc.clear_stored_data(date)
            if cleared:
                removed_factors.append(calc.factor_name)
        if verbosity > 0:
            print(f'Removed {len(removed_factors)} factors')

    @classmethod
    def iter_calculators(cls , all = False , selected_factors : list[str] | None = None , **kwargs) -> Generator[FactorCalculator , None , None]:
        '''iterate over calculators'''
        selected_factors = selected_factors or []
        if selected_factors:
            assert not all , \
                f'all ({all}) and selected_factors ({selected_factors}) cannot be supplied at once'
        return FactorCalculator.iter_calculators(all = all , selected_factors = selected_factors , **kwargs)

    @classmethod
    def update(cls , verbosity : int = 1 , groups_in_one_update : int | None = 100 , start : int | None = None , end : int | None = None) -> None:
        '''update factor data according'''
        self = cls()
        self.collect_jobs(start = start , end = end , all_factors = True , groups_in_one_update = groups_in_one_update)
        self.process_jobs(verbosity)
        self.update_factor_stats(start , end , all_factors = True)

    @classmethod
    def recalculate(cls , verbosity : int = 1 , groups_in_one_update : int | None = 100 , start : int | None = None , end : int | None = None) -> None:
        '''update factor data according'''
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        self = cls()
        self.collect_jobs(start = start , end = end , all_factors = True , overwrite = True , groups_in_one_update = groups_in_one_update)
        self.process_jobs(verbosity , overwrite = True)
        self.update_factor_stats(start , end , overwrite = True , all_factors = True)

    @classmethod
    def update_rollback(cls , rollback_date : int , verbosity : int = 1 , groups_in_one_update : int | None = 100) -> None:
        CALENDAR.check_rollback_date(rollback_date)
        self = cls()
        start = CALENDAR.td(rollback_date , 1)
        self.collect_jobs(start = start , all_factors = True , overwrite = True , groups_in_one_update = groups_in_one_update)
        self.process_jobs(verbosity , overwrite = True)
        self.update_factor_stats(start , overwrite = True , all_factors = True)

    @classmethod
    def update_fix(cls , factors : list[str] | None = None , verbosity : int = 1 , start : int | None = None , end : int | None = None) -> None:
        factors = factors or []
        self = cls()
        print(f'Fixing factors : {factors}')
        self.collect_jobs(selected_factors = factors , overwrite = True , start = start , end = end)
        self.process_jobs(verbosity , overwrite = True)
        self.update_factor_stats(start , end , overwrite = True , selected_factors = factors)

    @classmethod
    def eval_coverage(cls , all_factors = False , selected_factors : list[str] | None = None , **kwargs) -> pd.DataFrame:
        '''
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        '''
        selected_factors = selected_factors or []
        if not (all_factors or selected_factors or kwargs): 
            return pd.DataFrame()
        dfs : list[pd.DataFrame] = []

        for calc in cls.iter_calculators(all_factors , selected_factors , **kwargs):
            dates = calc.stored_dates()
            def load_fac(date : int):
                factor = calc.eval_factor(date)
                valid_count = factor.loc[:,calc.factor_name].notna().sum()
                return pd.DataFrame({'factor' : [calc.factor_name] , 'date' : [date] , 'valid_count' : [valid_count]})
            factor_coverage = parallel(load_fac , dates , method = 'threading')
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

UPDATE_JOBS = FactorUpdateJobManager()