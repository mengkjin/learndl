import numpy as np
import pandas as pd
import re

from abc import abstractmethod
from typing import Any , Literal , Type
from pathlib import Path

from ..util import StockFactor

from src.basic import CONF , CALENDAR , DB
from src.data import DATAVENDOR
from src.func.singleton import SingletonABCMeta
from src.func.parallel import parallel

class _FactorProperty:
    def __init__(self , method : str):
        assert method in dir(self) , f'{method} is not in {dir(self)}'
        self.method = method
        self.cache_values = {}

    def __get__(self,instance,owner) -> Any:
        if owner not in self.cache_values:
            self.cache_values[owner] = getattr(self , self.method)(owner)
        return self.cache_values[owner]

    def __set__(self,instance,value):
        raise AttributeError(f'{instance.__class__.__name__}.{self.method} is read-only attributes')

class _FactorPropertyStr(_FactorProperty):
    '''property of factor string'''
    def __get__(self,instance,owner) -> str:
        return super().__get__(instance,owner)

    def category0(self , owner) -> str:
        return CONF.Category1_to_Category0(owner.category1)

    def factor_name(self , owner) -> str:
        return owner.__qualname__

    def level(self , owner) -> str:
        '''level of the factor'''
        module : str = owner.__module__
        module = re.sub(r'[/.\\]' , '.' , module)
        level = re.search(r'.(level\d+).' , module)
        assert level is not None , f'level not found in {module}'
        return level.group(1)
    
    def file_name(self , owner) -> str:
        '''file name of the factor'''
        return '/'.join(Path(owner.__module__.split('level')[-1]).parts[1:]).removesuffix('.py')
    
    def factor_string(self , owner):
        '''full string of the factor'''
        return f'Factor {owner.level}/{owner.category0}/{owner.category1}/{owner.factor_name}'
    
class StockFactorCalculatorMeta(SingletonABCMeta):
    '''meta class of StockFactorCalculator'''
    ignore_names : list[str] = [
        'StockFactorCalculatorMeta' , 'StockFactorCalculator'
    ]
    registry : dict[str,Type['StockFactorCalculator'] | Any] = {}

    def __new__(cls, name, bases, dct):
        ''' 
        validate attribute of factor (init_date , description , category0 , category1)
        add subclass to registry
        '''
        new_cls = super().__new__(cls, name, bases, dct)
        if name not in cls.ignore_names:
            if dct.get('init_date' , -1) < CONF.FACTOR_INIT_DATE: 
                raise AttributeError(f'class {name} init_date should be later than {CONF.FACTOR_INIT_DATE} , but got {getattr(new_cls, "init_date" , -1)}')

            if not dct.get('description' , ''):
                raise AttributeError(f'class {name} description is not set')

            if not dct.get('category1' , ''):
                raise AttributeError(f'class {name} category1 is not set')

            CONF.Validate_Category(getattr(new_cls, 'category0' , '') , dct.get('category1' , ''))

            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            cls.registry[name] = new_cls
        return new_cls

class StockFactorCalculator(metaclass=StockFactorCalculatorMeta):
    '''base class of factor calculator'''
    init_date : int = -1
    category1 : Literal['quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''

    category0 = _FactorPropertyStr('category0')
    factor_name = _FactorPropertyStr('factor_name')
    level = _FactorPropertyStr('level')
    file_name = _FactorPropertyStr('file_name')
    factor_string = _FactorPropertyStr('factor_string')

    INIT_DATE = CONF.FACTOR_INIT_DATE
    CATEGORY0_SET = CONF.CATEGORY0_SET
    CATEGORY1_SET = CONF.CATEGORY1_SET
    FACTOR_CALENDAR = CALENDAR.td_within(INIT_DATE , step = CONF.UPDATE['step'])
    FACTOR_TARGET_DATES = CALENDAR.slice(FACTOR_CALENDAR , CONF.UPDATE['start'] , CONF.UPDATE['end'])
    UPDATE_MIN_VALID_COUNT_RELAX : int = 20
    UPDATE_MIN_VALID_COUNT_STRICT : int = 100
    UPDATE_RELAX_DATES : list[int] = []
    
    @classmethod
    def Calc(cls , date : int):
        return cls().calc_factor(date)
    
    @classmethod
    def Load(cls , date : int):
        return cls().load_factor(date)

    @classmethod
    def Loads(cls , start : int | None = None , end : int | None = None):
        dates = CALENDAR.slice(cls.stored_dates() , start , end)
        return DB.factor_load_multi(cls.factor_name , dates)

    @classmethod
    def Eval(cls , date : int):
        return cls().eval_factor(date)

    @classmethod
    def Factor(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , normalize = True , 
               fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
               weighted_whiten = False , order = ['fillna' , 'whiten' , 'winsor'] ,
               multi_thread = True , ignore_error = True , verbose = False):
        assert step % CONF.UPDATE['step'] == 0 , f'step {step} should be a multiple of {CONF.UPDATE["step"]}'
        dates = CALENDAR.slice(cls.FACTOR_CALENDAR , start , end)
        dates = dates[dates <= CALENDAR.updated()][::int(step/CONF.UPDATE['step'])]
        if len(dates) == 0: 
            return StockFactor(pd.DataFrame())
        calc = cls()
        def calculate_factor(date):
            df = calc.eval_factor(date , verbose = verbose)
            return df
        dfs = parallel(calculate_factor , dates , dates , method = multi_thread , ignore_error = ignore_error)
        factor = StockFactor(dfs , step = step)
        if normalize: 
            factor.normalize(fill_method , weighted_whiten , order , inplace = True)
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
        df = DB.factor_load(self.factor_name , date , verbose = False)
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
            if verbose: 
                print(f'{self.factor_name} at {date} recalculated')
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
            
            if df.empty: 
                df = pd.Series()
            else: 
                df : pd.Series | Any = df.rename(cls.factor_name).replace([np.inf , -np.inf] , np.nan)
                df = df[~df.index.duplicated(keep = 'first')]
                df = df.reindex(DATAVENDOR.secid(date))

            return df
        return new_calc_factor

    def __init_subclass__(cls, **kwargs):
        '''after subclassing , set calc_factor as wrapper'''
        super().__init_subclass__(**kwargs)
        setattr(cls , 'calc_factor' , cls.calc_factor_wrapper(cls.calc_factor))

    def __repr__(self):
        return f'{self.factor_name}(from{self.init_date},{self.category0},{self.category1})'
    
    def __call__(self , date : int):
        '''return factor value of a given date , calculate if not exist'''
        return self.calc_factor(date)

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
        if not overwrite and DB.factor_path(self.factor_name , date).exists(): 
            return False
        df = self.calc_factor(date)
        df = self.validate_value(df , date , strict = strict_validation)
        saved = DB.factor_save(df.rename(self.factor_name).to_frame() , self.factor_name , date , overwrite)
        return saved

    @classmethod
    def target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False , force = False):
        '''return list of target dates of factor data'''
        start = start if start is not None else cls.init_date
        end   = end   if end   is not None else 99991231
        dates = CALENDAR.slice(cls.FACTOR_CALENDAR if force else cls.FACTOR_TARGET_DATES , start , end)
        if not overwrite: 
            dates = CALENDAR.diffs(dates , cls.stored_dates())
        return dates
    
    @classmethod
    def stored_dates(cls): 
        '''return list of stored dates of factor data'''
        return DB.factor_dates(cls.factor_name)

    @classmethod
    def min_date(cls):
        return DB.factor_min_date(cls.factor_name)

    @classmethod
    def max_date(cls):
        return DB.factor_max_date(cls.factor_name)
    
    @classmethod
    def has_date(cls , date : int):
        return DB.factor_path(cls.factor_name , date).exists()

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

