import pandas as pd
import time

from typing import Any , Generator

from .factor_calc import FactorCalculator , PoolingCalculator

from src.proj import Logger
from src.basic import CONF , CALENDAR
from src.func.parallel import parallel

class PoolingFactorUpdater:
    """manager of factor update jobs"""
    _instance = None
    multi_thread : bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @classmethod
    def grouped_calculators(cls , all = True , selected_factors : list[str] | None = None , **kwargs) -> list[tuple[str , list[PoolingCalculator]]]:
        """group jobs by level and date"""
        calculators = list(cls.iter_calculators(all , selected_factors , **kwargs))
        groups : dict[str , list[PoolingCalculator]] = {}
        for calc in calculators:
            assert isinstance(calc , PoolingCalculator) , \
                f'{calc.factor_name} is not a pooling calculator'
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

        start_time = time.time()
        for level , calculators in cls.grouped_calculators(all , selected_factors , **kwargs):
            for calc in calculators:
                if verbosity > 0:
                    print(f'Updating {level} : {calc.factor_name} at {start} ~ {end}')
                calc.update_all(start = start , end = end , overwrite = overwrite , verbose = verbosity > 1)

            if timeout > 0 and time.time() - start_time > timeout * 3600:
                Logger.warning(f'Timeout: {timeout} hours reached, stopping update')
                Logger.warning(f'Terminated at level {level} , factor {calc.factor_name}')
                break

    @classmethod
    def iter_calculators(cls , all = True , selected_factors : list[str] | None = None , **kwargs) -> Generator[FactorCalculator , None , None]:
        '''iterate over calculators'''
        return FactorCalculator.iter_calculators(all , selected_factors , updatable = True , is_pooling = True , **kwargs)

    @classmethod
    def update(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        '''update factor data according'''
        cls.process_jobs(start = start , end = end , all = True , verbosity = verbosity , timeout = timeout)
        
    @classmethod
    def recalculate(cls , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        '''update factor data according'''
        assert start is not None and end is not None , 'start and end are required for recalculate factors'
        cls.process_jobs(start = start , end = end , all = True , overwrite = True , verbosity = verbosity , timeout = timeout)

    @classmethod
    def update_rollback(cls , rollback_date : int , verbosity : int = 1 , timeout : int = -1 , **kwargs) -> None:
        CALENDAR.check_rollback_date(rollback_date)
        start = CALENDAR.td(rollback_date , 1)
        cls.process_jobs(start = start , all = True , overwrite = True , verbosity = verbosity , timeout = timeout)
        
    @classmethod
    def update_fix(cls , factors : list[str] | None = None , verbosity : int = 1 , start : int | None = None , end : int | None = None , timeout : int = -1 , **kwargs) -> None:
        factors = factors or []
        print(f'Fixing factors : {factors}')
        cls.process_jobs(selected_factors = factors , overwrite = True , start = start , end = end , verbosity = verbosity , timeout = timeout)
        
    @classmethod
    def eval_coverage(cls , selected_factors : list[str] | None = None , **kwargs) -> pd.DataFrame:
        '''
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        '''
        dfs : list[pd.DataFrame] = []

        for calc in cls.iter_calculators(selected_factors = selected_factors , **kwargs):
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
