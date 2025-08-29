import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Any , Literal
from pathlib import Path

from ..util import StockFactor

from src.basic import CONF , PATH , CALENDAR
from src.data import DATAVENDOR
from src.func.singleton import SingletonABCMeta
from src.func.classproperty import classproperty_str
from src.func.parallel import parallel

class CategoryError(Exception): ...

class StockFactorCalculator(metaclass=SingletonABCMeta):
    init_date : int = -1
    category0 : Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative'] | Any
    category1 : Literal['quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''

    INIT_DATE = 20110101
    CATEGORY0_SET = ['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative']
    CATEGORY1_SET = {
        'fundamental' : ['quality' , 'growth' , 'value' , 'earning'] ,
        'analyst' : ['surprise' , 'coverage' , 'forecast' , 'adjustment'] ,
        'high_frequency' : ['hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity'] ,
        'behavior' : ['momentum' , 'volatility' , 'correlation' , 'liquidity'] ,
        'money_flow' : ['holding' , 'trading'] ,
        'alternative' : None
    }
    FACTOR_CALENDAR = CALENDAR.td_within(INIT_DATE , step = CONF.UPDATE_STEP)
    FACTOR_TARGET_DATES = CALENDAR.slice(FACTOR_CALENDAR , CONF.UPDATE_START , CONF.UPDATE_END)
    UPDATE_MIN_VALID_COUNT_RELAX  = 20
    UPDATE_MIN_VALID_COUNT_STRICT = 100
    UPDATE_RELAX_DATES : list[int] = []

    def __new__(cls , *args , **kwargs):
        return super().__new__(cls.validate_attrs())
    
    @classmethod
    def Calc(cls , date : int):
        return cls().calc_factor(date)
    
    @classmethod
    def Load(cls , date : int):
        return cls().load_factor(date)

    @classmethod
    def Loads(cls , start : int | None = None , end : int | None = None):
        dates = CALENDAR.slice(cls.stored_dates() , start , end)
        return PATH.factor_load_multi(cls.factor_name , dates)

    @classmethod
    def Eval(cls , date : int):
        return cls().eval_factor(date)

    @classmethod
    def Factor(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , normalize = True , 
               fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
               weighted_whiten = False , order = ['fillna' , 'whiten' , 'winsor'] ,
               multi_thread = True , ignore_error = True , verbose = False):
        assert step % CONF.UPDATE_STEP == 0 , f'step {step} should be a multiple of {CONF.UPDATE_STEP}'
        dates = CALENDAR.slice(cls.FACTOR_CALENDAR , start , end)
        dates = dates[dates <= CALENDAR.updated()][::int(step/CONF.UPDATE_STEP)]
        if len(dates) == 0: return StockFactor(pd.DataFrame())
        calc = cls()
        def calculate_factor(date):
            df = calc.eval_factor(date , verbose = verbose)
            return df
        dfs = parallel(calculate_factor , dates , dates , method = multi_thread , ignore_error = ignore_error)
        factor = StockFactor(dfs , step = step)
        if normalize: factor.normalize(fill_method , weighted_whiten , order , inplace = True)
        return factor

    @classmethod
    def FastTest(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , **kwargs):
        factor = cls.Factor(start , end , step , **kwargs)
        factor.fast_analyze()
        return factor

    @abstractmethod
    def calc_factor(self , date : int) -> pd.Series:
        '''calculate factor value , must have secid and factor_value / factor_name columns'''

    def load_factor(self , date : int):
        df = PATH.factor_load(self.factor_name , date , verbose = False)
        df = pd.Series() if df.empty else df.set_index('secid')[self.factor_name]
        return df

    def eval_factor(self , date : int , verbose : bool = False):
        try:
            df = self.load_factor(date)
        except Exception as e:
            print(f'{self.factor_name} at {date} error : {e}')
            df = pd.Series()
        if df.empty: 
            self.calc_and_deploy(date , overwrite = True)
            df = self.load_factor(date)
            assert not df.empty , f'factor {self.factor_name} is not calculated at {date}'
            if verbose: print(f'{self.factor_name} at {date} recalculated')
        return df
    
    @classmethod
    def calc_factor_wrapper(cls , raw_calc_factor):
        '''validate calculated factor value'''
        def new_calc_factor(self , date):
            date = int(date)
            if date < cls.init_date: 
                raise Exception(f'date should be >= {cls.init_date} for Factor {cls.factor_name} , but got {date}')
            
            df = raw_calc_factor(self , date)

            if not isinstance(df , pd.Series):  
                raise TypeError(f'calc_factor must return a Series , but got {type(df)} for factor {cls.factor_name}')
            
            if df.empty: df = pd.Series()
            else: 
                df = df.rename(cls.factor_name).replace([np.inf , -np.inf] , np.nan)
                df = df[~df.index.duplicated(keep = 'first')]
                df = df.reindex(DATAVENDOR.secid(date))

            return df
        return new_calc_factor

    def __init_subclass__(cls, **kwargs):
        '''after subclassing , set calc_factor as wrapper'''
        super().__init_subclass__(**kwargs)
        setattr(cls , 'calc_factor' , cls.calc_factor_wrapper(cls.calc_factor))
    
    @classproperty_str
    def factor_name(cls) -> str:
        return cls.__qualname__

    @classproperty_str
    def level(cls) -> str:
        '''level of the factor'''
        return 'level' + Path(cls.__module__.split('level')[-1]).parts[0]
    
    @classproperty_str
    def file_name(cls) -> str:
        '''file name of the factor'''
        return '/'.join(Path(cls.__module__.split('level')[-1]).parts[1:]).removesuffix('.py')
    
    @classproperty_str
    def factor_string(cls):
        return f'Factor {cls.level}/{cls.category0}/{cls.category1}/{cls.factor_name}'
    
    @classproperty_str
    def min_date(cls):
        return PATH.factor_min_date(cls.factor_name)
    
    @classproperty_str
    def max_date(cls):
        return PATH.factor_max_date(cls.factor_name)

    def __repr__(self):
        return f'{self.factor_name}(from{self.init_date},{self.category0},{self.category1})'
    
    def __call__(self , date : int):
        '''return factor value of a given date , calculate if not exist'''
        return self.calc_factor(date)
    
    @classmethod
    def validate_attrs(cls):
        '''
        validate attribute of factor
        init_date : must be greater than INIT_DATE(20070101)
        category0 : must be in CATEGORY0_SET([fundamental , analyst , high_frequency , behavior , money_flow , alternative])
        category1 : must be in CATEGORY1_SET[category0] if category1_list is not None , otherwise must be not None
            fundamental : quality , growth , value , earning
            analyst : surprise , coverage , forecast , adjustment
            high_frequency : hf_momentum , hf_volatility , hf_correlation , hf_liquidity
            behavior : momentum , volatility , correlation , liquidity
            money_flow : holding , trading
            alternative : None
        description : must be a non-empty string
        '''
        if cls.init_date < cls.INIT_DATE: 
            raise CategoryError(f'init_date should be later than {cls.INIT_DATE} , but got {cls.init_date}')

        cls.validate_category(cls.category0 , cls.category1)

        if not cls.description:
            raise CategoryError('description is not set')
        
        return cls

    @classmethod
    def validate_category(cls , category0 : str , category1 : str):
        if category0 not in cls.CATEGORY0_SET:
            raise CategoryError(f'category0 is should be in {cls.CATEGORY0_SET}, but got {category0}')
        
        if not category1:
            raise CategoryError('category1 is not set')
        
        if (category1_list := cls.CATEGORY1_SET[category0]):
            if category1 not in category1_list:
                raise CategoryError(f'category1 is should be in {category1_list}, but got {category1}')
        
    @classmethod
    def validate_value(cls , df : pd.Series , date : int , strict = False):
        '''validate factor value'''
        
        actual_valid_count = df.notna().sum()

        mininum_valid_count = cls.UPDATE_MIN_VALID_COUNT_STRICT
        if date in cls.UPDATE_RELAX_DATES or not strict:
            mininum_valid_count = cls.UPDATE_MIN_VALID_COUNT_RELAX

        if actual_valid_count < mininum_valid_count:
            raise ValueError(f'factor_value must have at least {mininum_valid_count} valid values , but got {actual_valid_count}')
        
        if np.isinf(df).any():
            raise ValueError(f'factor_value must not have infinite values , but got {np.isinf(df).sum()} infs')
        
        return df

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False) -> bool:
        '''store factor data after calculate'''
        if not overwrite and PATH.factor_path(self.factor_name , date).exists(): return False
        df = self.calc_factor(date)
        df = self.validate_value(df , date , strict = strict_validation)
        saved = PATH.factor_save(df.rename(self.factor_name).to_frame() , self.factor_name , date , overwrite)
        return saved

    @classmethod
    def target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False , force = False):
        '''return list of target dates of factor data'''
        start = start if start is not None else cls.init_date
        end   = end   if end   is not None else 99991231
        dates = CALENDAR.slice(cls.FACTOR_CALENDAR if force else cls.FACTOR_TARGET_DATES , start , end)
        if not overwrite: dates = CALENDAR.diffs(dates , cls.stored_dates())
        return dates
    
    @classmethod
    def stored_dates(cls): 
        '''return list of stored dates of factor data'''
        return PATH.factor_dates(cls.factor_name)
    
    @classmethod
    def has_date(cls , date : int):
        return PATH.factor_path(cls.factor_name , date).exists()

    @classmethod
    def collect_jobs(cls , start : int | None = None , end : int | None = None , 
                     overwrite = False , **kwargs):
        from src.res.factor.calculator.factor_update import UPDATE_JOBS
        UPDATE_JOBS.collect_jobs(start , end , overwrite = overwrite , **kwargs , factor_name = cls.factor_name)

    @classmethod
    def update(cls , verbosity : int = 1 , **kwargs):
        from src.res.factor.calculator.factor_update import UPDATE_JOBS
        cls.collect_jobs(overwrite = False , **kwargs)
        UPDATE_JOBS.process_jobs(verbosity , overwrite = False)

