import numpy as np
import pandas as pd
import polars as pl
import time

from dataclasses import dataclass
from typing import Any , Generator , Callable , Literal

from .factor_calc import FactorCalculator

from src.proj import Logger
from src.basic import CONF , CALENDAR 
from src.data import DATAVENDOR
from src.func.parallel import parallels
from src.func.singleton import SingletonMeta

__all__ = ['StockFactorUpdater' , 'MarketFactorUpdater' , 'RiskFactorUpdater' , 'PoolingFactorUpdater' , 'FactorStatsUpdater']

CATCH_ERRORS = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

class BaseFactorUpdater(metaclass=SingletonMeta):
    """manager of factor update jobs"""
    multi_thread : bool = True
    update_type : Literal['stock' , 'pooling' , 'risk' , 'market' , 'stats']
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @classmethod
    def calculators(cls , all = True , selected_factors : list[str] | None = None , updatable = True , **kwargs) -> list[FactorCalculator]:
        """get all calculators"""
        meta_type = None if cls.update_type == 'stats' else cls.update_type
        return list(FactorCalculator.iter_calculators(all , selected_factors , updatable = updatable , meta_type = meta_type , **kwargs))

    @classmethod
    def grouped_calculators(cls , all = True , selected_factors : list[str] | None = None , **kwargs) -> list[tuple[str , list[FactorCalculator]]]:
        """group jobs by level and date"""
        groups : dict[str , list[FactorCalculator]] = {}
        for calc in cls.calculators(all , selected_factors , **kwargs):
            if calc.level not in groups:
                groups[calc.level] = []
            groups[calc.level].append(calc)
        return sorted(groups.items() , key = lambda x: x[0])

    @classmethod
    def factors(cls) -> list[str]:
        """get all factors"""
        return [calc.factor_name for calc in cls.calculators()]

    @classmethod
    def preview_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """preview update jobs for all factors between start and end date"""
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)

        for calc in cls.calculators(all , selected_factors , **kwargs):
            print(f'{calc.level} : {calc.factor_name} at {start} ~ {end}')   

    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """process update jobs"""
        raise NotImplementedError(f'process_jobs is not implemented for {cls.__class__.__name__}')
               
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

        for calc in cls.calculators(selected_factors = selected_factors , **kwargs):
            dates = calc.stored_dates()
            def load_fac(date : int):
                factor = calc.eval_factor(date)
                valid_count = factor.loc[:,calc.factor_name].notna().sum()
                return pd.DataFrame({'factor' : [calc.factor_name] , 'date' : [date] , 'valid_count' : [valid_count]})
            calls = [(load_fac , (date , ) , {}) for date in dates]
            factor_coverage = parallels(calls , method = cls.multi_thread)
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
    jobs : list['OneUpdateJob'] = []
    multi_thread : bool = True
    update_type = 'stock'
    
    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.jobs)} jobs)'
    
    @dataclass  
    class OneUpdateJob:
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
    def append(cls , job : OneUpdateJob) -> None: 
        """append a job"""
        cls.jobs.append(job)
    @classmethod
    def grouped_jobs(cls) -> Generator[tuple[tuple[str , int] , list[OneUpdateJob]] , None , None]:
        """group jobs by level and date"""
        for level , date in cls.groups():
            yield (level , date) , cls.filter_jobs(cls.jobs , level , date)
    @staticmethod
    def filter_jobs(jobs : list[OneUpdateJob] , level : str , date : int) -> list[OneUpdateJob]:
        """filter jobs by level and date"""
        return [job for job in jobs if job.level == level and job.date == date]

    @classmethod
    def unfinished_factors(cls , date : int | None = None) -> dict[int , list[FactorCalculator]]:
        """get unfinished factors"""
        factors : dict[int , list[FactorCalculator]] = {}
        for calc in cls.calculators():
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

        for calc in cls.calculators(all , selected_factors , **kwargs):
            for date in calc.target_dates(start , end , overwrite = overwrite):
                cls.append(cls.OneUpdateJob(calc , date))

        if len(cls.jobs) == 0:
            print('There is no Stock Factor Update Jobs to Proceed...')
        else:
            levels , dates = cls.levels() , cls.dates()
            print(f'Finish Collecting {len(cls.jobs)} Stock Factor Update Jobs , levels: {levels} , dates: {min(dates)} ~ {max(dates)}')

    @classmethod
    def preview_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """preview update jobs for all factors between start and end date"""
        cls.collect_jobs(start , end , all , selected_factors , overwrite)

        for (level , date) , jobs in cls.grouped_jobs():
            for job in jobs:
                print(f'{level} : {job.factor_name} at {date}')
        

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

        start_time = time.time()
        for (level , date) , jobs in cls.grouped_jobs():
            DATAVENDOR.data_storage_control()
            keys = [job.factor_name for job in jobs]
            if verbosity > 1:
                if len(keys) > 10:
                    print(f'Updating {level} at {date} : {len(keys)} factors')
                else:
                    print(f'Updating {level} at {date} : {keys}')
            calls = [(job.do , (verbosity > 2 , overwrite) , {}) for job in jobs]
            parallels(calls , keys = keys , method = cls.multi_thread)
            failed_jobs = [job for job in jobs if not job.done]
            if verbosity > 0:
                print(f'Stock Factor Update of {level} at {date} Done: {len(jobs) - len(failed_jobs)} / {len(jobs)}')
                if failed_jobs: 
                    print(f'Failed Stock Factors: {[job.factor_name for job in failed_jobs]}')
            if auto_retry and failed_jobs:
                if verbosity > 0: 
                    print(f'Auto Retry Failed Stock Factors...')
                calls = [(job.do , (verbosity > 2 , overwrite) , {}) for job in failed_jobs]
                parallels(calls , method = len(failed_jobs) > 10)
                failed_again_jobs = [job for job in failed_jobs if not job.done]
                if failed_again_jobs:
                    print(f'Failed Stock Factors Again: {[job.factor_name for job in failed_again_jobs]}')
            if timeout and (time.time() - start_time) > timeout * 3600:
                Logger.warning(f'Timeout: {timeout} hours reached, stopping update')
                Logger.warning(f'Terminated at level {level} at date {date}')
                break
        [cls.jobs.remove(job) for job in jobs if job.done]

class MarketFactorUpdater(BaseFactorUpdater):
    """manager of factor update jobs"""
    multi_thread : bool = False
    update_type = 'market'
  
    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        """
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)
        
        grouped_calculators = cls.grouped_calculators(all , selected_factors , **kwargs)
        if len(grouped_calculators) == 0:
            print('There is no Market Factor Update Jobs to Proceed...')
            return
        else:
            n_jobs = len(grouped_calculators)
            levels = sorted(list(set([level for level , _ in grouped_calculators])))
            print(f'Finish Collecting {n_jobs} Market Factor Update Jobs , levels: {levels} , number of factors: {len(grouped_calculators)}')

        start_time = time.time()
        for level , calculators in grouped_calculators:
            for calc in calculators:
                if verbosity > 0:
                    print(f'Updating {level} Market Factor : {calc.factor_name} at {start} ~ {end}')
                calc.update_all_factors(start = start , end = end , overwrite = overwrite , verbose = verbosity > 1)

                if timeout and (time.time() - start_time) > timeout * 3600:
                    Logger.warning(f'Timeout: {timeout} hours reached, stopping update')
                    Logger.warning(f'Terminated at level {level} , factor {calc.factor_name}')
                    break  


class PoolingFactorUpdater(BaseFactorUpdater):
    """manager of factor update jobs"""
    multi_thread : bool = False
    update_type = 'pooling'
         
    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        """
        if end is None: 
            end = min(CALENDAR.updated() , CONF.Factor.UPDATE.end)
        
        grouped_calculators = cls.grouped_calculators(all , selected_factors , **kwargs)
        if len(grouped_calculators) == 0:
            print('There is no Pooling Factor Update Jobs to Proceed...')
            return
        else:
            n_jobs = len(grouped_calculators)
            levels = sorted(list(set([level for level , _ in grouped_calculators])))
            print(f'Finish Collecting {n_jobs} Pooling Factor Update Jobs , levels: {levels} , number of factors: {len(grouped_calculators)}')

        start_time = time.time()
        for level , calculators in grouped_calculators:
            for calc in calculators:
                if verbosity > 0:
                    print(f'Updating {level} Pooling Factor : {calc.factor_name} at {start} ~ {end}')
                calc.update_all_factors(start = start , end = end , overwrite = overwrite , verbose = verbosity > 1)

                if timeout and (time.time() - start_time) > timeout * 3600:
                    Logger.warning(f'Timeout: {timeout} hours reached, stopping update')
                    Logger.warning(f'Terminated at level {level} , factor {calc.factor_name}')
                    break  

class RiskFactorUpdater(BaseFactorUpdater):
    """manager of factor update jobs"""
    update_type = 'risk'
    
    @classmethod
    def preview_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        raise NotImplementedError(f'no job will be done for {cls.__class__.__name__}')

    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        """no job will be done for Risk Factor"""
        return        
    
class FactorStatsUpdater(BaseFactorUpdater):
    """manager of factor stats update jobs"""
    multi_thread : bool = False
    update_type = 'stats'
        
    @classmethod
    def process_jobs(cls , start : int | None = None , end : int | None = None , 
                            all = True , selected_factors : list[str] | None = None ,
                            overwrite = False , verbosity : int = 1 , **kwargs):
        """update all factor stats by year"""
        func_calls : dict[int , list[tuple[Callable , tuple[Any,...] , dict[str , Any] | None]]] = {}
        for calc in cls.calculators(all , selected_factors , **kwargs):
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
            parallels(calls , method = cls.multi_thread)
            total_calls += n_calls
            total_dates += n_dates

        print(f'Factor Stats Update Done: {len(func_calls)} Years , {total_calls} function calls , {total_dates} dates')           