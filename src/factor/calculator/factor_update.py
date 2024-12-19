import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Literal , Type

from src.basic import CONF
from src.data import DATAVENDOR
from src.func.singleton import singleton
from src.func.parallel import parallel

from .factor_calc import StockFactorCalculator

CATCH_ERRORS = (ValueError , TypeError)

@dataclass  
class FactorUpdateJob:
    calc : StockFactorCalculator
    date : int

    def __post_init__(self):
        self.done = False
        assert CONF.UPDATE_START <= self.date <= CONF.UPDATE_END , \
            f'date is should be between {CONF.UPDATE_START} and {CONF.UPDATE_END}, but got {self.date}'
    @property
    def level(self): return self.calc.level
    @property
    def factor_name(self): return self.calc.factor_name
    @property
    def sort_key(self): return (self.level , self.date , self.factor_name)
    def do(self , show_success : bool = False , catch_errors : tuple[Type[Exception],...] = () , overwrite = False):
        prefix = f'{self.calc.factor_string} at date {self.date}'
        try:
            self.done = self.calc.calc_and_deploy(self.date , overwrite = overwrite)
            if show_success:
                print(f'{prefix} ' + 'deploy successful' if self.done else 'already there')
        except catch_errors as e:
            print(f'{prefix} failed: {e}')
    
@singleton
class FactorUpdateJobManager:
    def __init__(self , multi_thread = True):
        self.jobs : list[FactorUpdateJob] = []
        self.multi_thread = multi_thread
    def __repr__(self):
        return f'FactorUpdateJobs({len(self)} jobs)'
    def __len__(self): return len(self.jobs)
    def levels(self): return np.unique([job.level for job in self.jobs])
    def dates(self): return np.unique([job.date for job in self.jobs])
    def to_dataframe(self):
        columns = ['level' , 'date' , 'factor']
        return pd.DataFrame([(job.level , job.date , job.factor_name) for job in self.jobs] , columns=columns).sort_values(by=columns)
    def filter(self , jobs : list[FactorUpdateJob] , level : str , date : int):
        return [job for job in jobs if job.level == level and job.date == date]
    def clear(self): self.jobs.clear()
    def sort(self): self.jobs.sort(key=lambda x: x.sort_key)
    def append(self , calc : StockFactorCalculator , date : int): 
        self.jobs.append(FactorUpdateJob(calc , date))
    def proceed(self , verbosity : int = 1 , ignore_error = False , overwrite = False):
        '''
        perform all update jobs

        verbosity : 
            0 : show only error
            1 : show error and success stats
            2 : show all
        ignore_error : if True , ignore some errors and continue
        overwrite : if True , overwrite existing data
        '''
        if len(self) == 0:
            print('There is no update jobs to proceed...')
            return
        levels , dates = self.levels() , self.dates()
        print(f'Finish collecting {len(self)} update jobs , levels: {levels} , dates: {min(dates)} ~ {max(dates)}')
        groups = sorted(set((job.level , job.date) for job in self.jobs))
        errors = CATCH_ERRORS if ignore_error else ()
        def do_job(job : FactorUpdateJob): job.do(verbosity > 1 , errors , overwrite)
        for level , date in groups:
            DATAVENDOR.data_storage_control()
            jobs = self.filter(self.jobs , level , date)
            parallel(do_job , jobs , method = self.multi_thread)
            if verbosity > 0:
                print(f'Factors of {level} at {date} done: {sum(job.done for job in jobs)} / {len(jobs)}')
            [self.jobs.remove(job) for job in jobs if job.done]

UPDATE_JOBS = FactorUpdateJobManager()