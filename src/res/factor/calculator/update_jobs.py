from __future__ import annotations
import numpy as np
import polars as pl

from abc import ABC , abstractmethod
from datetime import datetime
from typing import Any , Generator , TypedDict

from src.proj import Base , Logger
from src.proj.util.functional.parallel import parallel
from src.data import DATAVENDOR

from .factor_calc import FactorCalculator

__all__ = [
    'UpdateJobDate' , 'UpdateJobAll' , 'UpdateJobStats' ,
    'JobChunkPayload' , 'JobChunkReport' , 'run_job_chunk_payload',
]

CATCH_ERRORS : tuple[type[Exception],...] = (ValueError , TypeError , pl.exceptions.ColumnNotFoundError)

class JobResultEntry(TypedDict):
    job_type: str
    factor_name: str
    done: bool
    date: int | None
    year: int | None
    stats_type: str | None

class JobChunkReport(TypedDict):
    name: str
    results: list[JobResultEntry]
    ok: int
    total: int

class JobChunkPayload(TypedDict):
    name: str
    specs: list[dict[str, Any]]
    multithreading: bool
    timeout: float

class BaseUpdateJob(ABC):
    """base job class"""
    def __init__(self , calc : FactorCalculator , overwrite : bool = False , * , indent : int = 1 , vb_level : Any = 2):
        self.calc : FactorCalculator | Any = calc
        self.level = calc.level
        self.factor_name = calc.factor_name
        self.overwrite = overwrite
        self.done = False

        self.indent = indent
        self.vb_level = vb_level

    def __repr__(self):
        return self.factor_name

    def __call__(self , **kwargs) -> None:
        return self.go(**kwargs)

    def regenerate(self) -> None:
        """Restore ``FactorCalculator`` after ``degenerate`` cleared the reference for pickling."""
        if self.calc is None:
            self.calc = FactorCalculator.get(self.factor_name)

    def degenerate(self) -> None:
        """Drop calculator instance before sending jobs across process boundaries."""
        self.calc = None

    def degenrate(self) -> None:
        """Alias for :meth:`degenerate` (historic spelling)."""
        self.degenerate()

    def match(self , **kwargs) -> bool:
        """check if the job matches the given attributes"""
        for key , value in kwargs.items():
            if getattr(self , key) != value:
                return False
        return True

    def report_key(self) -> tuple[Any, ...]:
        """Key used to merge :class:`JobChunkReport` entries back into this job."""
        raise NotImplementedError

    def to_spec(self) -> dict[str, Any]:
        """Pickle-friendly job description (no ``FactorCalculator`` instance)."""
        raise NotImplementedError

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


def _job_from_spec(spec: dict[str, Any]) -> BaseUpdateJob:
    calc = FactorCalculator.get(spec['factor_name'])
    job_type = spec['job_type']
    if job_type == 'date':
        return UpdateJobDate(calc , spec['date'] , spec['overwrite'])
    if job_type == 'all':
        return UpdateJobAll(calc , spec.get('start') , spec.get('end') , spec['overwrite'])
    if job_type == 'stats':
        return UpdateJobStats(
            calc , spec['stats_type'] , spec['year'] ,
            np.array(spec['year_dates'] , dtype=np.int64) ,
            spec['overwrite'] ,
        )
    raise ValueError(f'Unknown job_type in spec: {job_type!r}')

def run_job_chunk_payload(payload: JobChunkPayload) -> JobChunkReport:
    """
    Process-pool entry point (module-level, picklable).

    Rebuilds jobs in the worker, runs :meth:`BaseUpdateJobList.process`, and returns
    a serializable report for the parent process.
    """
    from src.proj.util.catcher import MPOutputCatcher

    jobs = [_job_from_spec(s) for s in payload['specs']]
    chunk = BaseUpdateJobList(
        payload['name'] , jobs ,
        multithreading=payload['multithreading'] ,
        timeout=payload['timeout'],
    )
    try:
        return chunk.process(in_worker=True)
    finally:
        MPOutputCatcher.export_current_task(payload['name'])


class BaseUpdateJobList(Base.BoundLogger):
    """base update job list class"""
    def __init__(self , name : str , jobs : list[BaseUpdateJob] | None = None , * ,
        multithreading : bool = False , timeout : float = -1 , vb_level : Any = 1 , indent : int = 0 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.name = name
        self.jobs : list[BaseUpdateJob] = jobs or []
        self.multithreading = multithreading
        self.timeout = timeout

    def __len__(self) -> int:
        return len(self.jobs)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __call__(self , **kwargs) -> JobChunkReport:
        return self.process(**kwargs)

    def __getitem__(self , index : int) -> BaseUpdateJob:
        return self.jobs[index]

    def __setitem__(self , index : int , job : BaseUpdateJob) -> None:
        self.jobs[index] = job

    def __delitem__(self , index : int) -> None:
        del self.jobs[index]

    def __iter__(self) -> Generator[BaseUpdateJob , None , None]:
        for job in self.jobs:
            yield job

    def __contains__(self , job : BaseUpdateJob) -> bool:
        return job in self.jobs

    def __repr__(self) -> str:
        return f'UpdateJobList({self.jobs})'

    def with_name(self , name : str) -> BaseUpdateJobList:
        self.name = name
        return self

    def with_kwargs(
        self , multithreading : bool | None = None ,
        vb_level : Any | None = None , indent : int | None = None , timeout : float | None = None ,
    ) -> BaseUpdateJobList:
        if multithreading is not None:
            self.multithreading = multithreading
        if timeout is not None:
            self.timeout = timeout
        self.set_vb(vb_level , indent)
        return self

    def jobs_dict(self , failed_only : bool = True) -> dict[BaseUpdateJob , BaseUpdateJob]:
        return {job: job for job in self.jobs if not failed_only or not job.done}

    def filter_jobs(self , **kwargs) -> BaseUpdateJobList:
        new_name = f'{self.name}({kwargs})'
        return BaseUpdateJobList(new_name , [job for job in self.jobs if job.match(**kwargs)])

    def split_by(self , split_keys : list[str] , **kwargs) -> dict[tuple , BaseUpdateJobList]:
        split_values = sorted(set(tuple(getattr(job , key) for key in split_keys) for job in self.jobs))
        ret = {}
        for split_value in split_values:
            split_group = dict(zip(split_keys , split_value))
            ret[split_value] = self.filter_jobs(**split_group).with_kwargs(**kwargs)
        return ret

    def levels(self) -> np.ndarray:
        return np.unique([job.level for job in self.jobs])

    def clear(self) -> None:
        self.jobs.clear()

    def append(self , job : BaseUpdateJob) -> None:
        self.jobs.append(job)

    def extend(self , jobs : list[BaseUpdateJob]) -> None:
        self.jobs.extend(jobs)

    def sort_jobs(self) -> None:
        self.jobs.sort(key=lambda x: x.sort_key())

    def degenerate_jobs(self) -> None:
        for job in self.jobs:
            job.degenerate()

    def degenrate_jobs(self) -> None:
        self.degenerate_jobs()

    def regenerate_jobs(self) -> None:
        for job in self.jobs:
            job.regenerate()

    def to_payload(self) -> JobChunkPayload:
        """Serializable payload for :func:`run_job_chunk_payload` (no calculator objects)."""
        return JobChunkPayload(
            name=self.name ,
            specs=[job.to_spec() for job in self.jobs] ,
            multithreading=self.multithreading ,
            timeout=self.timeout ,
        )

    def apply_report(self , report: JobChunkReport | None) -> None:
        """Merge worker-process results into in-memory ``job.done`` flags."""
        if not report:
            return
        by_key = {
            (e['job_type'] , e['factor_name'] , e.get('date') , e.get('year') , e.get('stats_type')): e['done']
            for e in report['results']
        }
        for job in self.jobs:
            if job.report_key() in by_key:
                job.done = bool(by_key[job.report_key()])

    def build_report(self) -> JobChunkReport:
        results: list[JobResultEntry] = []
        for job in self.jobs:
            entry: JobResultEntry = {
                'job_type': job.to_spec()['job_type'] ,
                'factor_name': job.factor_name ,
                'done': bool(job.done) ,
                'date': None ,
                'year': None ,
                'stats_type': None ,
            }
            spec = job.to_spec()
            entry['date'] = spec.get('date')
            entry['year'] = spec.get('year')
            entry['stats_type'] = spec.get('stats_type')
            results.append(entry)
        ok = sum(1 for r in results if r['done'])
        return JobChunkReport(name=self.name , results=results , ok=ok , total=len(results))

    def process(self , * , multithreading : bool | None = None ,
                timeout : float | None = None , in_worker : bool = False) -> JobChunkReport:
        """
        Run all jobs in this chunk (in-process). Returns a report for parent merge.

        When ``in_worker`` is False and jobs were degenerated for pickling, call
        :meth:`regenerate_jobs` before this only if specs were not used to rebuild.
        """
        multithreading = self.multithreading if multithreading is None else multithreading
        timeout = self.timeout if timeout is None else timeout

        DATAVENDOR.data_storage_control()
        self.logger.stdout(f'Updating {self.name} : ' + (f'{len(self)} factors' if len(self) > 10 else str(self.jobs)))

        if not in_worker:
            self.regenerate_jobs()
        self.sort_jobs()
        start_time = datetime.now()
        parallel(self.jobs_dict(failed_only=False) , method=multithreading , timeout=timeout , indent=self.indent + 1)
        remaining_timeout = (timeout - (datetime.now() - start_time).total_seconds() / 3600) if timeout > 0 else 0
        report = self.build_report()
        self.logger.success(f'Stock Factor Update of {self.name} : {report["ok"]} / {report["total"]}')
        failed_jobs = self.jobs_dict(failed_only=True)
        if failed_jobs and timeout > 0 and remaining_timeout > 0:
            self.logger.alert1(f'Failed Stock Factors: {list(failed_jobs.keys())}')
            self.logger.stdout('Auto Retry Failed Stock Factors...')
            parallel(failed_jobs , method=multithreading , timeout=remaining_timeout , indent=self.indent + 1)
            report = self.build_report()
            if [job for job in failed_jobs.values() if not job.done]:
                self.logger.alert1(f'Failed Stock Factors Again: {list(failed_jobs.keys())}')
            else:
                self.logger.success(f'All {len(failed_jobs)} Failed Stock Factors are Processed Successfully!')
        return report

    @classmethod
    def merge_jobs(cls , job_lists : list[BaseUpdateJobList] , name : str = 'merged_jobs' , **kwargs) -> BaseUpdateJobList:
        jobs = [job for job_list in job_lists for job in job_list.jobs]
        return BaseUpdateJobList(name , jobs , **kwargs)

class UpdateJobDate(BaseUpdateJob):
    def __init__(self , calc : FactorCalculator , date : int , overwrite : bool = False , * , indent : int = 1 , vb_level : Any = 2):
        super().__init__(calc , overwrite , indent = indent , vb_level = vb_level)
        self.date = date

    def __repr__(self):
        return f'{self.factor_name}({self.date})'

    def dates(self) -> np.ndarray:
        return np.array([self.date])

    def sort_key(self) -> Any:
        return (self.level , self.date , self.factor_name)

    def report_key(self) -> tuple[Any, ...]:
        return ('date' , self.factor_name , self.date , None , None)

    def to_spec(self) -> dict[str, Any]:
        return {
            'job_type': 'date' ,
            'factor_name': self.factor_name ,
            'level': self.level ,
            'overwrite': self.overwrite ,
            'date': int(self.date) ,
        }

    def go(self , **kwargs) -> None:
        self.calc.set_vb(self.vb_level , self.indent)
        self.regenerate()
        self.done = self.calc.update_day_factor(self.date , overwrite=self.overwrite , catch_errors=CATCH_ERRORS)

    def preview(self) -> None:
        Logger.stdout(f'{self.level} : {self.factor_name} at {self.date}' , indent = self.indent , vb_level = self.vb_level)


class UpdateJobAll(BaseUpdateJob):
    def __init__(self , calc : FactorCalculator , start : int | None = None , end : int | None = None ,
                 overwrite : bool = False , * , indent : int = 1 , vb_level : Any = 2):
        super().__init__(calc , overwrite , indent = indent , vb_level = vb_level)
        self.target_dates = calc.target_dates(start , end , overwrite=overwrite)
        self.start = start
        self.end = end

    def __repr__(self):
        return f'{self.factor_name}({self.start}-{self.end})'

    def dates(self) -> np.ndarray:
        return self.target_dates

    def sort_key(self) -> Any:
        return (self.level , self.factor_name)

    def report_key(self) -> tuple[Any, ...]:
        return ('all' , self.factor_name , None , None , None)

    def to_spec(self) -> dict[str, Any]:
        return {
            'job_type': 'all' ,
            'factor_name': self.factor_name ,
            'level': self.level ,
            'overwrite': self.overwrite ,
            'start': self.start ,
            'end': self.end ,
        }

    def go(self , **kwargs) -> None:
        self.calc.set_vb(self.vb_level , self.indent)
        self.regenerate()
        self.calc.update_all_factors(start=self.start , end=self.end , overwrite=self.overwrite)
        self.done = True

    def preview(self) -> None:
        Logger.stdout(f'Updating {self.level} : {self.factor_name} from {self.start} to {self.end}' , indent = self.indent , vb_level = self.vb_level)


class UpdateJobStats(BaseUpdateJob):
    def __init__(self , calc : FactorCalculator , stats_type : Base.lit.FactorStatsPeriod ,
                 year : int , dates : np.ndarray , overwrite : bool = False , * , indent : int = 1 , vb_level : Any = 2):
        super().__init__(calc , overwrite , indent = indent , vb_level = vb_level)
        self.stats_type = stats_type
        self.year = year
        self.year_dates = dates[dates // 10000 == year]

    def __repr__(self):
        return f'{self.factor_name}({self.year}-{self.stats_type})'

    def dates(self) -> np.ndarray:
        return self.year_dates

    def sort_key(self) -> Any:
        return (self.year , self.factor_name)

    def report_key(self) -> tuple[Any, ...]:
        return ('stats' , self.factor_name , None , self.year , self.stats_type)

    def to_spec(self) -> dict[str, Any]:
        return {
            'job_type': 'stats' ,
            'factor_name': self.factor_name ,
            'level': self.level ,
            'overwrite': self.overwrite ,
            'stats_type': self.stats_type ,
            'year': int(self.year) ,
            'year_dates': self.year_dates.astype(int).tolist() ,
        }

    def go(self , **kwargs) -> None:
        self.calc.set_vb(self.vb_level , self.indent)
        self.regenerate()
        if self.stats_type == 'daily':
            self.calc.update_daily_stats(self.dates() , overwrite=self.overwrite)
        elif self.stats_type == 'weekly':
            self.calc.update_weekly_stats(self.dates() , overwrite=self.overwrite)
        else:
            raise ValueError(f'Invalid stats type: {self.stats_type}')
        self.done = True

    def preview(self) -> None:
        Logger.stdout(f'{self.level} : {self.factor_name} - {self.stats_type} at year {self.year}' , indent = self.indent , vb_level = self.vb_level)
