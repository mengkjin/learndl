import time
import numpy as np
import pandas as pd

from dataclasses import dataclass
from threading import Lock

from src.basic import CONF , CALENDAR
from src.data import DATAVENDOR
from src.func.singleton import singleton
from src.func.parallel import parallel

from .factor_calc import StockFactorCalculator
from .hierarchy import StockFactorHierarchy

CATCH_ERRORS = (ValueError , TypeError)

@dataclass  
class FactorUpdateJob:
    calc : StockFactorCalculator
    date : int

    def __post_init__(self):
        self.done = False
        assert CONF.UPDATE_START <= self.date <= CONF.UPDATE_END , \
            f'date is should be between {CONF.UPDATE_START} and {CONF.UPDATE_END}, but got {self.date}'
        assert self.date in StockFactorCalculator.FACTOR_TARGET_DATES , \
            f'date {self.date} is not in StockFactorCalculator.FACTOR_TARGET_DATES'
        
    @property
    def level(self): return self.calc.level
    @property
    def factor_name(self): return self.calc.factor_name
    @property
    def sort_key(self): return (self.level , self.date , self.factor_name)
    def do(self , show_success : bool = False , overwrite = False):
        prefix = f'{self.calc.factor_string} at date {self.date}'
        try:
            self.done = self.calc.calc_and_deploy(self.date , overwrite = overwrite)
            if show_success:
                print(f'{prefix} ' + 'deploy successful' if self.done else 'already there')
        except CATCH_ERRORS as e:
            print(f'{prefix} failed: {e}')
    
@singleton
class FactorUpdateJobManager:
    def __init__(self , multi_thread = True):
        self.jobs : list[FactorUpdateJob] = []
        self.lock = Lock()
        self.multi_thread = multi_thread
    def __repr__(self):
        return f'FactorUpdateJobs({len(self)} jobs)'
    def __len__(self): return len(self.jobs)
    def levels(self): return np.unique([job.level for job in self.jobs])
    def dates(self): return np.unique([job.date for job in self.jobs])
    def groups(self): return sorted(set((job.level , job.date) for job in self.jobs))
    def to_dataframe(self):
        columns = ['level' , 'date' , 'factor']
        return pd.DataFrame([(job.level , job.date , job.factor_name) for job in self.jobs] , columns=columns).sort_values(by=columns)

    def clear(self): self.jobs.clear()
    def sort(self): self.jobs.sort(key=lambda x: x.sort_key)
    def append(self , job : FactorUpdateJob): self.jobs.append(job)
    def grouped_jobs(self):
        for level , date in self.groups():
            yield (level , date) , self.filter_jobs(self.jobs , level , date)
    @staticmethod
    def filter_jobs(jobs : list[FactorUpdateJob] , level : str , date : int):
        return [job for job in jobs if job.level == level and job.date == date]
    def unfinished_factors(self , date : int | None = None):
        if date is None: 
            factors = {}
            for date in StockFactorCalculator.FACTOR_TARGET_DATES:
                if f := self.unfinished_factors(date): factors[date] = f
            return factors
        else:
            if date not in StockFactorCalculator.FACTOR_TARGET_DATES:
                return []
            else:
                hier = StockFactorHierarchy()
                factors = [calc for calc in hier.iter_instance() if not calc.has_date(date)]
                return factors

    def collect_jobs(self , start : int | None = None , end : int | None = None , 
                     all_factors = False , selected_factors : list[str] = [] ,
                     overwrite = False , groups_in_one_update : int | None = None , **kwargs):
        '''
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        '''
        self.clear()
        if not (all_factors or selected_factors or kwargs): return self
        if end is None: end = min(CALENDAR.updated() , CONF.UPDATE_END)

        for calc in self.iter_calculators(all_factors , selected_factors , **kwargs):
            dates = calc.target_dates(start , end , overwrite = overwrite)
            [self.append(FactorUpdateJob(calc , d)) for d in dates]

        if len(self) == 0:
            print('There is no update jobs to proceed...')
        else:
            if groups_in_one_update is not None:
                groups = self.groups()[:groups_in_one_update]
                self.jobs = [job for level , date in groups for job in self.filter_jobs(self.jobs , level , date)]
            levels , dates = self.levels() , self.dates()
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} : Finish collecting {len(self)} update jobs , levels: {levels} , dates: {min(dates)} ~ {max(dates)}')
        return self
    
    def proceed(self , verbosity : int = 1 , overwrite = False):
        '''
        perform all update jobs

        verbosity : 
            0 : show only error
            1 : show error and success stats
            2 : show all
        overwrite : if True , overwrite existing data
        '''
        if len(self) == 0: return

        def do_job(job : FactorUpdateJob): 
            job.do(verbosity > 1 , overwrite)

        for (level , date) , jobs in self.grouped_jobs():
            DATAVENDOR.data_storage_control()
            parallel(do_job , jobs , method = self.multi_thread)
            if verbosity > 0:
                failed_factors = [job.factor_name for job in jobs if not job.done]
                print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} : Factors of {level} at {date} done: {len(jobs) - len(failed_factors)} / {len(jobs)}')
                if failed_factors:
                    print(f'Failed factors: {failed_factors}')
        [self.jobs.remove(job) for job in jobs if job.done]
    
    @classmethod
    def iter_calculators(cls , all_factors = False , selected_factors : list[str] = [] , **kwargs):
        if selected_factors:
            assert not all_factors , \
                f'all_factors ({all_factors}) and selected_factors ({selected_factors}) cannot be supplied at once'
        hier = StockFactorHierarchy()
        if all_factors:
            return hier.iter_instance()
        elif selected_factors:
            return (calc for calc in hier.iter_instance() if calc.factor_name in selected_factors)
        else:
            return hier.iter_instance(**kwargs)

    @classmethod
    def update(cls , verbosity : int = 1 , groups_in_one_update : int | None = 100 , start : int | None = None , end : int | None = None):
        '''update factor data according'''
        self = cls()
        self.collect_jobs(start = start , end = end , all_factors = True , groups_in_one_update = groups_in_one_update)
        self.proceed(verbosity)

    @classmethod
    def update_fix(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None):
        self = cls()
        fix_factors = cls.fix_factors()
        print(f'Fixing factors : {fix_factors}')
        self.collect_jobs(selected_factors = fix_factors , overwrite = True , start = start , end = end)
        self.proceed(verbosity , overwrite = True)

    @classmethod
    def eval_coverage(cls , all_factors = False , selected_factors : list[str] = [] , **kwargs):
        '''
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        '''
        
        if not (all_factors or selected_factors or kwargs): return pd.DataFrame()
        dfs : list[pd.DataFrame] = []

        for calc in cls.iter_calculators(all_factors , selected_factors , **kwargs):
            dates = calc.stored_dates()
            def load_fac(date : int):
                try:
                    factor = calc.load_factor(date)
                except Exception as e:
                    print(f'load factor {calc.factor_string} at {date} failed: {e}')
                    calc.calc_and_deploy(date , overwrite = True)
                    factor = calc.load_factor(date)
                valid_count = factor.notna().sum()
                return pd.DataFrame({'factor' : [calc.factor_string] , 'date' : [date] , 'valid_count' : [valid_count]})
            factor_coverage = parallel(load_fac , dates , method = 'threading')
            dfs.extend(list(factor_coverage.values()))

        df = pd.concat(dfs)
        agg = pd.concat([
            df.groupby(by='factor').mean()['valid_count'].rename('mean') ,
            df.groupby(by='factor').min()['valid_count'].rename('min') ,
            df.groupby(by='factor').max()['valid_count'].rename('max') ,
            df.groupby(by='factor').std()['valid_count'].rename('std') ,
        ] , axis = 1)
        df.to_excel('factor_coverage.xlsx' , sheet_name='full coverage')
        agg.to_excel('factor_coverage_agg.xlsx' , sheet_name='coverage_agg_stats')
        return df

    @classmethod
    def fix_factors(cls):
        return []

UPDATE_JOBS = FactorUpdateJobManager()