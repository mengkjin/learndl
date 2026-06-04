from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from datetime import datetime , timedelta
from functools import cached_property
from typing import Any , Generator , Literal

from src.proj import CALENDAR , Const , BaseMeta , BaseClass
from src.proj.util.functional.parallel import parallel
from src.proj.util.filesys.ttl_cache import DiskTTLCache
from src.data import DATAVENDOR

from .factor_calc import FactorCalculator , WeightedPoolingCalculator
from .update_jobs import (
    BaseUpdateJobList , UpdateJobDate , UpdateJobAll , UpdateJobStats ,
    run_job_chunk_payload ,
)

__all__ = ['StockFactorUpdater' , 'MarketFactorUpdater' , 'AffiliateFactorUpdater' , 'PoolingFactorUpdater' , 'FactorStatsUpdater']

CATCH_ERRORS = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

MAX_PROCESS_NUM = Const.Factor.UPDATE.get(f'update_groups_multiprocessing/max_workers')
MIN_PROCESS_NUM = Const.Factor.UPDATE.get(f'update_groups_multiprocessing/min_workers')

class BaseFactorUpdater(BaseClass.BoundLogger , metaclass=BaseMeta.Singleton):
    """manager of factor update jobs"""
    update_type : Literal['stock' , 'pooling' , 'affiliate' , 'market' , 'stats']

    def __init__(self , * , indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.jobs = BaseUpdateJobList(f'{self.name} Factor Jobs')
    
    def __repr__(self):
        n_jobs = len(self.jobs) if self.jobs is not None else 0
        return f'{self.name}({n_jobs} jobs)'

    @cached_property
    def groups_multiprocessing(self) -> bool:
        return Const.Factor.UPDATE.get(f'update_groups_multiprocessing/{self.update_type}')

    @cached_property
    def jobs_multithreading(self) -> bool:
        return not self.groups_multiprocessing and Const.Factor.UPDATE.get(f'update_jobs_multithreading')

    @property
    def name(self) -> str:
        """name of the updater"""
        return self.__class__.__name__

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

    def collect_jobs(self , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """
        collect FactorCalculator jobs for all factors between start and end date
        """
        self.jobs.clear()
        record_entry = DiskTTLCache.get('daily_update', f'{self.name}_empty_jobs')
        if record_entry.valid_value:
            self.logger.skipping(f'There is no {self.name} Jobs to Proceed ({record_entry.time_str}) ...' , idt = 1 , vb = 1)
            return
        
        if end is None: 
            end = min(CALENDAR.updated() , Const.Factor.UPDATE.end)

        for calc in self.calculators(all , selected_factors , **kwargs):
            if self.update_type in ['affiliate']:
                ...
            elif self.update_type in ['market' , 'pooling']:
                target_dates = calc.target_dates(start , end , overwrite = overwrite)
                if len(target_dates) > 0:
                    self.jobs.append(UpdateJobAll(calc , start , end , overwrite , indent = self.indent + 2 , vb_level = self.vb_level + 2))
            elif self.update_type in ['stock']:
                target_dates = calc.target_dates(start , end , overwrite = overwrite)
                for date in target_dates:
                    self.jobs.append(UpdateJobDate(calc , date , overwrite , indent = self.indent + 2 , vb_level = self.vb_level + 4))
            elif self.update_type == 'stats':
                target_dates = calc.stats_target_dates(start , end , overwrite)
                for stats_type , dates in target_dates.items():
                    for year in np.unique(dates // 10000):
                        self.jobs.append(UpdateJobStats(calc , stats_type , year , dates , overwrite , indent = self.indent + 2 , vb_level = self.vb_level + 4))
            else:
                raise ValueError(f'Invalid update type: {self.update_type}')
        self.jobs.sort_jobs()

        if self.jobs:
            self.logger.success(f'Collecting {len(self.jobs)} Jobs for {self.name}' , idt = 1 , vb = 1)
        else:
            self.logger.skipping(f'There is no {self.name} Jobs to Proceed...' , idt = 1 , vb = 1)
            record_entry.put(True , ttl_hours = 8)
        
    def before_process_jobs(self , start : int | None = None , end : int | None = None , 
                            all = True , selected_factors : list[str] | None = None ,
                            overwrite = False , **kwargs) -> None:
        """before process jobs"""
        for calc in self.calculators(all , selected_factors , **kwargs):
            if isinstance(calc , WeightedPoolingCalculator):
                calc.drop_pooling_weight(after = start , overwrite = overwrite)

    def preview_jobs(self , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , **kwargs) -> None:
        """preview update jobs for all factors between start and end date"""
        self.collect_jobs(start , end , all , selected_factors , overwrite = overwrite , vb_level = 'never')
        [job.preview() for job in self.jobs]

    def after_process_jobs(self , **kwargs) -> None:
        """after process jobs"""
        failed_jobs = [job for job in self.jobs if not job.done]
        if failed_jobs:
            if len(failed_jobs) <= 10:
                self.logger.alert1(f'Remaining Failed Jobs: {failed_jobs}', idt = 1 , vb = 1)
            else:
                self.logger.alert1(f'Remaining {len(failed_jobs)} Jobs Failed: [{str(failed_jobs[:10])[:-1]},...]', idt = 1 , vb = 1)
        elif len(self.jobs) > 0:
            self.logger.success(f'All {len(self.jobs)} Jobs are Processed Successfully!' , idt = 1 , vb = 1)

    def process_jobs(self , start : int | None = None , end : int | None = None , 
                     all = True , selected_factors : list[str] | None = None ,
                     overwrite = False , timeout : float = -1 , **kwargs) -> None:
        """
        update update jobs for all factors between start and end date
        default behavior is to collect jobs first to find all FactorCalculators then update one by one
        timeout : timeout for processing jobs in hours , if <= 0 , no timeout
        """
        self.collect_jobs(start , end , all , selected_factors , overwrite)
        self.before_process_jobs(start , end , all , selected_factors , overwrite)

        timeout_time = datetime.now() + timedelta(hours=timeout) if timeout > 0 else None
        for level , level_jobs in self.leveled_jobs():
            remaining_timeout = -1 if timeout_time is None else (timeout_time - datetime.now()).total_seconds() / 3600
            self.process_level_jobs(level , level_jobs , timeout = remaining_timeout)
            if timeout_time and datetime.now() >= timeout_time:
                break
        
        self.after_process_jobs()

    def leveled_jobs(self) -> Generator[tuple[int , BaseUpdateJobList]]:
        """group jobs by level"""
        for level in self.jobs.levels():
            yield level , self.jobs.filter_jobs(level = level)
    
    def grouped_jobs(self , level : int , level_jobs : BaseUpdateJobList , **kwargs) -> dict[str , BaseUpdateJobList]:
        """
        group jobs by specified rules of a given level
        eg. for stock factor updater, level = 'level1' , factor_name = 'factor1'
        groups = sorted(set((job.level , job.factor_name) for job in cls.jobs))
        for level , factor_name in groups:
            yield (level , factor_name) , cls.filter_jobs(cls.jobs , level = level , factor_name = factor_name)
        """
        raise NotImplementedError(f'grouped_jobs is not implemented for {self.name}')

    def process_level_jobs(self ,
        level : int , level_jobs : BaseUpdateJobList , 
        timeout : float = -1 , **kwargs
    ) -> None:
        self.logger.stdout(f'Updating level {level} : ' + (f'{len(level_jobs)} factors' if len(level_jobs) > 8 else str(level_jobs)) , idt = 2 , vb = 2)
        assert not self.groups_multiprocessing or not self.jobs_multithreading , 'groups_multiprocessing and multithreading cannot be used together'
        group_kwargs = {
            'multithreading' : self.jobs_multithreading ,
            'indent' : self.indent + 3 ,
            'vb_level' : self.vb_level + 4 ,
            'timeout' : timeout ,
        }
        group_jobs = self.grouped_jobs(level , level_jobs , **group_kwargs)
        if self.groups_multiprocessing:
            # Strip calculators before pickling; worker rebuilds from specs and returns a report.
            for chunk in group_jobs.values():
                chunk.degenerate_jobs()
            process_inputs = {name: (run_job_chunk_payload , [chunk.to_payload()]) for name , chunk in group_jobs.items()}
            DATAVENDOR.data_storage_control()
            results = parallel(
                process_inputs , method = 'process' , timeout = timeout , 
                indent = self.indent + 2 , capture_mp_output = True ,
            )
            for name , report in results.items():
                group_jobs[name].apply_report(report)
            level_jobs.regenerate_jobs()
        else:
            timeout_time = datetime.now() + timedelta(hours=timeout) if timeout > 0 else None
            for chunk in group_jobs.values():
                self.logger.stdout(f'Updating {chunk.name} : ' + (f'{len(chunk)} factors' if len(chunk) > 10 else str(chunk)) , idt = 3 , vb = 4)
                remaining_timeout = -1 if timeout_time is None else (timeout_time - datetime.now()).total_seconds() / 3600
                chunk.process(timeout = remaining_timeout)
                if timeout_time and datetime.now() >= timeout_time:
                    break

    @classmethod
    def update(cls , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        """update pooling factor data according"""
        updater = cls()
        updater.logger.note('Update since last update!')
        updater.process_jobs(start = start , end = end , all = True , timeout = timeout)

    @classmethod
    def recalculate(cls , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        """recalculate factors between start and end date"""
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        updater = cls()
        updater.logger.note('Recalculate all factors!')
        updater.process_jobs(start = start , end = end , all = True , overwrite = True , timeout = timeout)

    @classmethod
    def rollback(cls , rollback_date : int , timeout : int = -1 , **kwargs) -> None:
        CALENDAR.check_rollback_date(rollback_date)
        start = int(CALENDAR.td(rollback_date , 1))
        updater = cls()
        updater.logger.note(f'Rollback from {rollback_date}!')
        updater.process_jobs(start = start , all = True , overwrite = True , timeout = timeout)
        
    @classmethod
    def fix(cls , factors : list[str] , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        assert factors , 'factors are required for fix'
        factors = [factor for factor in cls.factors() if factor in factors]
        if factors:
            updater = cls()
            updater.logger.note(f'Fixing {factors} from {start} to {end}!')
            updater.process_jobs(selected_factors = factors , overwrite = True , start = start , end = end , timeout = timeout)

    def eval_coverage(self , selected_factors : list[str] | None = None , **kwargs) -> pd.DataFrame:
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
            
        for calc in self.calculators(selected_factors = selected_factors , **kwargs):
            dates = calc.stored_dates()
            calls = {date:(load_fac , {'calc' : calc , 'date' : date}) for date in dates}
            factor_coverage = parallel(calls , method = self.jobs_multithreading)
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

def _split_one_jobs(jobs : BaseUpdateJobList , num_chunks : int = MIN_PROCESS_NUM , **kwargs) -> list[BaseUpdateJobList]:
    """split one job list into multiple job lists"""
    jobs_per_chunk = int(np.ceil(len(jobs) / num_chunks))
    start = 0
    chunks = []
    for i in range(num_chunks):
        chunk_name = f'index={i+1}/{num_chunks}'
        chunk_jobs = BaseUpdateJobList(chunk_name , jobs.jobs[start:start+jobs_per_chunk] , **kwargs)
        chunks.append(chunk_jobs)
        start += jobs_per_chunk
    return chunks

def _split_multiple_jobs(all_jobs : dict[tuple , BaseUpdateJobList] , num_chunks : int = MAX_PROCESS_NUM , chunk_step_ratio : float = 0.9 , **kwargs) -> dict[tuple[tuple , tuple] , BaseUpdateJobList]:
    assert len(all_jobs) > 1 , f'all_jobs must be a dictionary with at least 2 keys, {len(all_jobs)} keys found'
    num_jobs = len(all_jobs)
    num_chunks = min(len(all_jobs) , num_chunks)
    chunk_ratios = np.power(chunk_step_ratio , np.arange(num_chunks))
    chunk_ratios /= chunk_ratios.sum()
    chunk_job_nums = (np.array(chunk_ratios) * num_jobs).astype(int)
    while sum(chunk_job_nums) < num_jobs:
        chunk_job_nums += 1
    while sum(chunk_job_nums) >= num_jobs + num_chunks:
        chunk_job_nums -= 1
    i = -1
    while sum(chunk_job_nums) > num_jobs:
        chunk_job_nums[i] -= 1
        i -= 1
    ret = {}
    start = 0
    all_jobs_keys = list(all_jobs.keys())
    all_jobs_values = list(all_jobs.values())
    for i , chunk_job_num in enumerate(chunk_job_nums):
        keys = all_jobs_keys[start:start+chunk_job_num]
        values = all_jobs_values[start:start+chunk_job_num]
        if len(chunk_job_num) == 0:
            continue
        else:
            chunk_keys= keys[0] , keys[-1]
        chunk_jobs = BaseUpdateJobList.merge_jobs(values).with_name(f'chunk{i+1}/{num_chunks}').with_kwargs(**kwargs)
        ret[chunk_keys] = chunk_jobs
        start += chunk_job_num
    return ret

class StockFactorUpdater(BaseFactorUpdater):
    """manager of factor update jobs"""
    update_type = 'stock'

    def grouped_jobs(self , level : int , level_jobs : BaseUpdateJobList , **kwargs) -> dict[str , BaseUpdateJobList]:
        """group jobs by level and date"""
        date_jobs = level_jobs.split_by(split_keys = ['date'])
        dates = np.unique([date for (date,) in date_jobs.keys()]).astype(int)
        assert len(date_jobs) == len(dates) , f'date_jobs and dates are not the same length, {len(date_jobs)} != {len(dates)}'
        ret = {}
        if dates.size == 1:
            (date , ) , jobs = list(date_jobs.items())[0]
            chunks = _split_one_jobs(jobs , MIN_PROCESS_NUM , **kwargs)
            for chunk in chunks:
                chunk_name = f'{self.name}(level={level},date={date},{chunk.name})'
                ret[chunk_name] = chunk
        elif dates.size > 1:
            chunks = _split_multiple_jobs(date_jobs , MAX_PROCESS_NUM , **kwargs)
            for ((date_start,) , (date_end,)) , chunk_jobs in chunks.items():
                if date_start == date_end:
                    chunk_datestr = str(date_start)
                else:
                    chunk_datestr = f'{date_start}-{date_end}'
                chunk_name = f'{self.name}(level={level},date={chunk_datestr})'
                ret[chunk_name] = chunk_jobs
        return ret
class MarketFactorUpdater(BaseFactorUpdater):
    """manager of market factor update jobs"""
    update_type = 'market'

    def grouped_jobs(self , level : int , level_jobs : BaseUpdateJobList , **kwargs) -> dict[str , BaseUpdateJobList]:
        """group jobs by level and date"""
        ret = {}
        for job in level_jobs:
            chunk_jobs = BaseUpdateJobList(f'{self.name}(level={level},factor={job.factor_name})' , [job] , **kwargs)
            ret[chunk_jobs.name] = chunk_jobs
        return ret

class PoolingFactorUpdater(BaseFactorUpdater):
    """manager of pooling factor update jobs"""
    update_type = 'pooling'

    def grouped_jobs(self , level : int , level_jobs : BaseUpdateJobList , **kwargs) -> dict[str , BaseUpdateJobList]:
        """group jobs by level and factor_name"""
        ret = {}
        for job in level_jobs:
            chunk_jobs = BaseUpdateJobList(f'{self.name}(level={level},factor={job.factor_name})' , [job] , **kwargs)
            ret[chunk_jobs.name] = chunk_jobs
        return ret
        
class AffiliateFactorUpdater(BaseFactorUpdater):
    """manager of affiliate factor update jobs"""
    update_type = 'affiliate'

    def grouped_jobs(self , level : int , level_jobs : BaseUpdateJobList , **kwargs) -> dict[str , BaseUpdateJobList]:
        """group jobs by level and factor_name"""
        ret = {}
        for job in level_jobs:
            chunk_jobs = BaseUpdateJobList(f'{self.name}(level={level},factor={job.factor_name})' , [job] , **kwargs)
            ret[chunk_jobs.name] = chunk_jobs
        return ret

class FactorStatsUpdater(BaseFactorUpdater):
    """manager of factor stats update jobs"""
    update_type = 'stats'

    @cached_property
    def groups_multiprocessing(self) -> bool:
        """factor stats update does not support multiprocessing, always return False"""
        return False

    @cached_property
    def jobs_multithreading(self) -> int:
        """factor stats update does not support multithreading, always return True for jobs_multithreading"""
        return True

    def grouped_jobs(self , level : int , level_jobs : BaseUpdateJobList , **kwargs) -> dict[str , BaseUpdateJobList]:
        """group jobs by level and date"""
        assert not self.groups_multiprocessing , 'factor stats update does not support multiprocessing'
        year_jobs = level_jobs.split_by(split_keys = ['year'])
        ret = {}
        for (year,) , jobs in year_jobs.items():
            chunk_jobs = jobs.with_name(f'{self.name}(level={level},year={year})').with_kwargs(**kwargs)
            ret[chunk_jobs.name] = chunk_jobs
        return ret