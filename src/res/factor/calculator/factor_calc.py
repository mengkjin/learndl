import numpy as np
import pandas as pd
import re

from abc import abstractmethod
from typing import Any , Callable , Literal , Type
from pathlib import Path

from ..util import StockFactor

from src.proj import Logger
from src.basic import CONF , CALENDAR , DB
from src.data import DATAVENDOR
from src.func.singleton import SingletonABCMeta
from src.func.parallel import parallel

__all__ = [
    'StockFactorCalculator' ,
    'QualityFactor' , 'GrowthFactor' , 'ValueFactor' , 'EarningFactor' , 'SurpriseFactor' , 'CoverageFactor' , 'ForecastFactor' , 'AdjustmentFactor' ,
    'HfMomentumFactor' , 'HfVolatilityFactor' , 'HfCorrelationFactor' , 'HfLiquidityFactor' ,
    'MomentumFactor' , 'VolatilityFactor' , 'CorrelationFactor' , 'LiquidityFactor' , 'HoldingFactor' , 'TradingFactor'
]

class _FactorProperty:
    """get any property of a factor calculator , cached"""
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
    """property of factor string"""
    def __get__(self,instance,owner) -> str:
        return super().__get__(instance,owner)

    def category0(self , owner) -> str:
        return CONF.Factor.STOCK.cat1_to_cat0(owner.category1)

    def factor_name(self , owner) -> str:
        return owner.__qualname__

    def level(self , owner) -> str:
        """level of the factor"""
        module : str = owner.__module__
        module = re.sub(r'[/.\\]' , '.' , module)
        level = re.search(r'.(level\d+).' , module)
        assert level is not None , f'level not found in {module}'
        return level.group(1)
    
    def file_name(self , owner) -> str:
        """file name of the factor"""
        return '/'.join(Path(owner.__module__.split('level')[-1]).parts[1:]).removesuffix('.py')
    
    def factor_string(self , owner) -> str:
        """full string of the factor"""
        return f'Factor {owner.level}/{owner.category0}/{owner.category1}/{owner.factor_name}'
    

def _calc_factor_wrapper(calc_factor : Callable[['StockFactorCalculator',int],pd.Series]) -> Callable[['StockFactorCalculator',int],pd.Series]:
    """
    check and modify before and after factor value calculation
    before:
        1. date should be >= init_date
    after:
        1. calc_factor must return a Series
        2. rename to factor_name
        3. replace infinite values to nan
        4. remove duplicate secid
        5. reindex to secid(date)
    """
    def wrapper(instance : 'StockFactorCalculator' , date : int):
        date = int(date)
        if date < instance.init_date: 
            raise Exception(f'date should be >= {instance.init_date} for Factor {instance.factor_name} , but got {date}')
        
        df = calc_factor(instance , date)

        if not isinstance(df , pd.Series):  
            raise TypeError(f'calc_factor must return a Series , but got {type(df)} for factor {instance.factor_name}')
        
        if df.empty: 
            df = pd.Series()
        else: 
            df : pd.Series | Any = df.rename(instance.factor_name).replace([np.inf , -np.inf] , np.nan)
            df = df[~df.index.duplicated(keep = 'first')]
            df = df.reindex(DATAVENDOR.secid(date))

        return df
    return wrapper

class _StockFactorCalculatorMeta(SingletonABCMeta):
    """meta class of StockFactorCalculator"""
    registry : dict[str,Type['StockFactorCalculator'] | Any] = {}

    def __new__(cls, name, bases, dct):
        """ 
        validate attribute of factor (init_date , description , category0 , category1)
        only if all abstract methods are implemented , then add subclass to registry
        also wrap calc_factor with _calc_factor_wrapper
        """
        new_cls = super().__new__(cls, name, bases, dct)
        abstract_methods = getattr(new_cls , '__abstractmethods__' , None)
        if not abstract_methods:
            if dct.get('init_date' , -1) < CONF.Factor.UPDATE.init_date: 
                raise AttributeError(f'class {name} init_date should be later than {CONF.Factor.UPDATE.init_date} , but got {getattr(new_cls, "init_date" , -1)}')

            if not dct.get('description' , ''):
                raise AttributeError(f'class {name} description is not set')

            assert 'category1' not in dct , f'cannot set category1 in {name} , use the corresponding BaseClass in StockFactorCalculator'
            assert 'category0' not in dct , f'cannot set category0 in {name} , use the corresponding BaseClass in StockFactorCalculator'

            category0 , category1 = getattr(new_cls, 'category0' , ''), getattr(new_cls, 'category1' , '')
            if not category1:
                raise AttributeError(f'class {name} category1 is not set')
            if not category0:
                raise AttributeError(f'class {name} category0 is not set')

            CONF.Factor.STOCK.validate_categories(category0 , category1)

            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            
            setattr(new_cls , 'calc_factor' , _calc_factor_wrapper(getattr(new_cls , 'calc_factor')))

            cls.registry[name] = new_cls
        return new_cls

class StockFactorCalculator(metaclass=_StockFactorCalculatorMeta):
    """base class of factor calculator"""
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

    INIT_DATE = CONF.Factor.UPDATE.init_date
    CATEGORY0_SET = CONF.Factor.STOCK.category0
    CATEGORY1_SET = CONF.Factor.STOCK.category1
    FACTOR_CALENDAR = CALENDAR.td_within(INIT_DATE , step = CONF.Factor.UPDATE.step)
    FACTOR_TARGET_DATES = CALENDAR.slice(FACTOR_CALENDAR , CONF.Factor.UPDATE.start , CONF.Factor.UPDATE.end)
    UPDATE_MIN_VALID_COUNT_RELAX : int = 20
    UPDATE_MIN_VALID_COUNT_STRICT : int = 100
    UPDATE_RELAX_DATES : list[int] = []

    def __repr__(self):
        return f'{self.factor_name.capitalize()} Calculator(init_date={self.init_date},category0={self.category0},category1={self.category1},description={self.description})'
    
    def __call__(self , date : int):
        """return factor value of a given date , calculate if not exist"""
        return self.calc_factor(date)

    @abstractmethod
    def calc_factor(self , date : int) -> pd.Series:
        """calculate factor value , must have secid and factor_value / factor_name columns"""

    def load_factor(self , date : int) -> pd.Series:
        """load factor value of a given date"""
        df = DB.factor_load(self.factor_name , date , verbose = False)
        df = pd.Series() if df.empty else df.set_index('secid').loc[:,self.factor_name]
        return df

    def eval_factor(self , date : int , verbose : bool = False) -> pd.Series:
        """get factor value of a given date , load if exist , calculate if not exist"""
        try:
            df = self.load_factor(date)
        except Exception as e:
            Logger.error(f'{self.factor_name} at {date} error : {e}')
            df = pd.Series()
        if df.empty: 
            self.calc_and_deploy(date , overwrite = True)
            df = self.load_factor(date)
            assert not df.empty , f'factor {self.factor_name} is not calculated at {date}'
            if verbose: 
                print(f'{self.factor_name} at {date} recalculated')
        return df

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False) -> bool:
        """store factor data after calculate"""
        if not overwrite and DB.factor_path(self.factor_name , date).exists(): 
            return False
        df = self.calc_factor(date)
        df = self.validate_value(df , date , strict = strict_validation)
        saved = DB.factor_save(df.rename(self.factor_name).to_frame() , self.factor_name , date , overwrite)
        return saved

    @classmethod
    def Calc(cls , date : int) -> pd.Series:
        """calculate factor value of a given date"""
        return cls().calc_factor(date)
    
    @classmethod
    def Load(cls , date : int) -> pd.Series:
        """load factor value of a given date"""
        return cls().load_factor(date)

    @classmethod
    def Loads(cls , start : int | None = None , end : int | None = None) -> pd.DataFrame:
        """load factor values of a given date range"""
        dates = CALENDAR.slice(cls.stored_dates() , start , end)
        return DB.factor_load_multi(cls.factor_name , dates)

    @classmethod
    def Eval(cls , date : int) -> pd.Series:
        """get factor value of a given date , load if exist , calculate if not exist"""
        return cls().eval_factor(date)

    @classmethod
    def Factor(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , normalize = True , 
               fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
               weighted_whiten = False , order = ['fillna' , 'whiten' , 'winsor'] ,
               multi_thread = True , ignore_error = True , verbose = False) -> StockFactor:
        """get factor values of a given date range , load if exist , calculate if not exist"""
        assert step % CONF.Factor.UPDATE.step == 0 , f'step {step} should be a multiple of {CONF.Factor.UPDATE.step}'
        dates = CALENDAR.slice(cls.FACTOR_CALENDAR , start , end)
        dates = dates[dates <= CALENDAR.updated()][::int(step/CONF.Factor.UPDATE.step)]
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
    def FastTest(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , **kwargs) -> StockFactor:
        """fast test of factor values of a given date range"""
        factor = cls.Factor(start , end , step , **kwargs)
        factor.fast_analyze()
        return factor

    @classmethod
    def validate_value(cls , df : pd.Series , date : int , strict = False) -> pd.Series:
        """
        validate factor value of a given date
        1. valid count should be >= UPDATE_MIN_VALID_COUNT_STRICT or UPDATE_MIN_VALID_COUNT_RELAX
        2. factor_value must not have infinite values
        """
        actual_valid_count = df.notna().sum()

        mininum_valid_count = cls.UPDATE_MIN_VALID_COUNT_STRICT
        if date in cls.UPDATE_RELAX_DATES or not strict:
            mininum_valid_count = cls.UPDATE_MIN_VALID_COUNT_RELAX

        if actual_valid_count < mininum_valid_count:
            raise ValueError(f'factor_value must have at least {mininum_valid_count} valid values , but got {actual_valid_count}')
        
        if np.isinf(df).any():
            raise ValueError(f'factor_value must not have infinite values , but got {np.isinf(df).sum()} infs')
        
        return df

    @classmethod
    def target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False , force = False) -> np.ndarray:
        """returntarget dates of factor updates"""
        start = start if start is not None else cls.init_date
        end   = end   if end   is not None else 99991231
        dates = CALENDAR.slice(cls.FACTOR_CALENDAR if force else cls.FACTOR_TARGET_DATES , start , end)
        if not overwrite: 
            dates = CALENDAR.diffs(dates , cls.stored_dates())
        return dates
    
    @classmethod
    def stored_dates(cls) -> np.ndarray:
        """return stored dates of factor data"""
        return DB.factor_dates(cls.factor_name)

    @classmethod
    def min_date(cls) -> int:
        """return minimum date of stored factor data"""
        return DB.factor_min_date(cls.factor_name)

    @classmethod
    def max_date(cls) -> int:
        """return maximum date of stored factor data"""
        return DB.factor_max_date(cls.factor_name)
    
    @classmethod
    def has_date(cls , date : int) -> bool:
        """check if factor data exists for a given date"""
        return DB.factor_path(cls.factor_name , date).exists()

    @classmethod
    def collect_jobs(cls , start : int | None = None , end : int | None = None , 
                     overwrite = False , **kwargs) -> None:
        """collect update jobs of this factor within a given date range"""
        from src.res.factor.calculator.factor_update import UPDATE_JOBS
        UPDATE_JOBS.collect_jobs(start , end , overwrite = overwrite , **kwargs , factor_name = cls.factor_name)

    @classmethod
    def update(cls , verbosity : int = 1 , **kwargs) -> None:
        """update factor data to the latest date"""
        from src.res.factor.calculator.factor_update import UPDATE_JOBS
        cls.collect_jobs(overwrite = False , **kwargs)
        UPDATE_JOBS.process_jobs(verbosity , overwrite = False)

class QualityFactor(StockFactorCalculator):
    """Factor Calculator of category0: fundamental , category1: quality"""
    category1 = 'quality'

class GrowthFactor(StockFactorCalculator):
    """Factor Calculator of category0: fundamental , category1: growth"""
    category1 = 'growth'

class ValueFactor(StockFactorCalculator):
    """Factor Calculator of category0: fundamental , category1: value"""
    category1 = 'value'

class EarningFactor(StockFactorCalculator):
    """Factor Calculator of category0: fundamental , category1: earning"""
    category1 = 'earning'

class SurpriseFactor(StockFactorCalculator):
    """Factor Calculator of category0: analyst , category1: surprise"""
    category1 = 'surprise'

class CoverageFactor(StockFactorCalculator):
    """Factor Calculator of category0: analyst , category1: coverage"""
    category1 = 'coverage'

class ForecastFactor(StockFactorCalculator):
    """Factor Calculator of category0: analyst , category1: forecast"""
    category1 = 'forecast'

class AdjustmentFactor(StockFactorCalculator):
    """Factor Calculator of category0: analyst , category1: adjustment"""
    category1 = 'adjustment'

class HfMomentumFactor(StockFactorCalculator):
    """Factor Calculator of category0: high_frequency , category1: hf_momentum"""
    category1 = 'hf_momentum'

class HfVolatilityFactor(StockFactorCalculator):
    """Factor Calculator of category0: high_frequency , category1: hf_volatility"""
    category1 = 'hf_volatility'

class HfCorrelationFactor(StockFactorCalculator):
    """Factor Calculator of category0: high_frequency , category1: hf_correlation"""
    category1 = 'hf_correlation'

class HfLiquidityFactor(StockFactorCalculator):
    """Factor Calculator of category0: high_frequency , category1: hf_liquidity"""
    category1 = 'hf_liquidity'

class MomentumFactor(StockFactorCalculator):
    """Factor Calculator of category0: behavior , category1: momentum"""
    category1 = 'momentum'

class VolatilityFactor(StockFactorCalculator):
    """Factor Calculator of category0: behavior , category1: volatility"""
    category1 = 'volatility'

class CorrelationFactor(StockFactorCalculator):
    """Factor Calculator of category0: behavior , category1: correlation"""
    category1 = 'correlation'

class LiquidityFactor(StockFactorCalculator):
    """Factor Calculator of category0: behavior , category1: liquidity"""
    category1 = 'liquidity'

class HoldingFactor(StockFactorCalculator):
    """Factor Calculator of category0: money_flow , category1: holding"""
    category1 = 'holding'

class TradingFactor(StockFactorCalculator):
    """Factor Calculator of category0: money_flow , category1: trading"""
    category1 = 'trading'
