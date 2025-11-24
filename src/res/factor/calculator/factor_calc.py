import numpy as np
import pandas as pd
import re , traceback

from importlib import import_module
from abc import abstractmethod
from typing import Any , Callable , Literal , Type , Generator
from pathlib import Path

from ..util import StockFactor

from src.proj import Logger , PATH
from src.basic import CONF , CALENDAR , DB
from src.data import DATAVENDOR
from src.func.singleton import SingletonABCMeta
from src.func.parallel import parallel

__all__ = [
    'FactorCalculator' , 'StockFactorCalculator' , 'MarketFactorCalculator' , 'AffiliateFactorCalculator' ,
    'QualityFactor' , 'GrowthFactor' , 'ValueFactor' , 'EarningFactor' , 'SurpriseFactor' , 'CoverageFactor' , 'ForecastFactor' , 'AdjustmentFactor' ,
    'HfMomentumFactor' , 'HfVolatilityFactor' , 'HfCorrelationFactor' , 'HfLiquidityFactor' ,
    'MomentumFactor' , 'VolatilityFactor' , 'CorrelationFactor' , 'LiquidityFactor' , 'HoldingFactor' , 'TradingFactor' ,
    'StyleFactor' , 'SellsideFactor' , 'MarketEventFactor' , 'WeightedPoolingCalculator' , 'NonlinearPoolingCalculator'
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
        return '/'.join(Path(owner.__module__.split('level')[-1].replace('.' , '/')).parts[1:]).removesuffix('.py')
    
    def factor_string(self , owner) -> str:
        """full string of the factor"""
        return f'Factor {owner.level}/{owner.category0}/{owner.category1}/{owner.factor_name}'

class _FactorPropertyBool(_FactorProperty):
    """property of boolean"""
    def __init__(self , method : Literal['is_pooling' , 'is_market']):
        self.method = method

    def __get__(self,instance,owner) -> bool:
        value = getattr(self , self.method)(owner)
        return value

    def is_pooling(self , owner) -> bool:
        return owner.meta_type == 'pooling'

    def is_market(self , owner) -> bool:
        return owner.meta_type == 'market'

class _FactorMetaType:
    """meta class of factor"""
    def __get__(self,instance,owner) -> Literal['market' , 'stock' , 'affiliate' , 'pooling']:
        return CONF.Factor.STOCK.cat0_to_meta(owner.category0)

class _FactorDBSrc:
    """db source of factor"""
    def __get__(self,instance,owner) -> Literal['stock_factor' , 'market_factor']:
        meta_type = getattr(owner, 'meta_type')
        if meta_type == 'market':
            return 'market_factor'
        elif meta_type in ['stock' , 'affiliate' , 'pooling']:
            return 'stock_factor'
        else:
            raise ValueError(f'undefined meta type: {meta_type} for factor {owner.__qualname__}')

class _FactorDBKey:
    """db source of factor"""
    def __get__(self,instance,owner) -> str:
        return getattr(owner, 'factor_name')

class _FactorCalendar:
    """calendar of factor"""
    _dates = CALENDAR.td_within(CONF.Factor.UPDATE.init_date)

    def __init__(self , method : Literal['update' , 'calendar'] = 'calendar'):
        self.method = method
    
    def __get__(self,instance,owner) -> np.ndarray:

        init_date = getattr(owner, 'init_date')
        final_date = getattr(owner, 'final_date')
        update_step = getattr(owner, 'update_step') if self.method == 'update' else 1
        
        assert update_step > 0 , f'update_step should be greater than 0 for {owner.__qualname__} , but got {update_step}'
        dates = self._dates[::update_step]
        dates = dates[dates >= init_date]
        dates = dates[dates <= final_date]
        if self.method == 'update':
            dates = dates[(dates >= CONF.Factor.UPDATE.start) & (dates <= CONF.Factor.UPDATE.end)]
        return dates

class _FactorStoredDates:
    """min date of factor"""
    def __init__(self , method : Literal['min' , 'max'] = 'min'):
        self.method = method
    def __get__(self,instance,owner) -> int:
        return getattr(owner , f'get_{self.method}_date')()

def _calc_factor_wrapper(calc_factor : Callable[['FactorCalculator',int],pd.Series | pd.DataFrame]) -> Callable[['FactorCalculator',int], pd.DataFrame]:
    """
    check and modify before and after factor value calculation
    before:
        1. date should be >= init_date
    after:
        1. raw calc_factor must return a Series or DataFrame
        2. rename to factor_name
        3. replace infinite values to nan
        4. remove duplicate secid
        5. reindex to secid(date)
    """
    def wrapper(instance : 'FactorCalculator' , date : int):
        date = int(date)
        
        if date < instance.init_date: 
            raise Exception(f'date should be >= {instance.init_date} for Factor {instance.factor_name} , but got {date}')
        
        df = calc_factor(instance , date)

        if not isinstance(df , (pd.Series , pd.DataFrame)):  
            raise TypeError(f'calc_factor must return a Series or DataFrame , but got {type(df)} for factor {instance.factor_name}')
        
        if isinstance(df , pd.Series): 
            df = df.rename(instance.factor_name).to_frame()
            
        if len(df.index.names) > 1 or df.index.name:
            df = df.reset_index()
    
        assert instance.factor_name in df.columns , f'factor_name {instance.factor_name} not found in calc_factor result: {df.columns}'
        if instance.meta_type == 'market':
            assert 'date' in df.columns , f'date not found in calc_factor result for market factor: {df.columns}'
            df = df.drop_duplicates(subset = ['date'] , keep = 'first').sort_values('date')
        elif instance.meta_type in ['stock' , 'affiliate' , 'pooling']:
            assert 'secid' in df.columns , f'secid not found in calc_factor result for stock factor: {df.columns}'
            df = df.drop_duplicates(subset = ['secid'] , keep = 'first').set_index('secid').reindex(DATAVENDOR.secid(date)).reset_index(drop = False)
        else:
            raise ValueError(f'undefined meta type: {instance.meta_type}')

        for col in df.columns:
            if df[col].dtype != 'O':
                df[col] = df[col].mask(np.isinf(df[col]), np.nan)
        df = df.reset_index(drop = True)
        return df
    return wrapper

class _FactorCalculatorMeta(SingletonABCMeta):
    """meta class of StockFactorCalculator"""

    registry : dict[str,Type['FactorCalculator'] | Any] = {}
    definition_imported : bool = False

    def __new__(cls, name, bases, dct):
        """ 
        validate attribute of factor calculator (init_date , description , category0 , category1 , meta_type)
        only if all abstract methods are implemented , then add subclass to registry
        also wrap calc_factor with _calc_factor_wrapper
        """
        new_cls = super().__new__(cls, name, bases, dct)
        abstract_methods = getattr(new_cls , '__abstractmethods__' , None)
        registered = getattr(new_cls , 'registered' , True)
        if not abstract_methods and registered:
            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'

            if getattr(new_cls, "init_date" , -1) < CONF.Factor.UPDATE.init_date: 
                raise AttributeError(f'class {name} init_date should be later than {CONF.Factor.UPDATE.init_date} , but got {getattr(new_cls, "init_date" , -1)}')

            if not getattr(new_cls, "description" , -1):
                raise AttributeError(f'class {name} description is not set')

            if issubclass(new_cls , StockFactorCalculator):
                assert getattr(new_cls, 'meta_type') == 'stock' , f'{name} must be a stock factor'
            elif issubclass(new_cls , MarketFactorCalculator):
                assert getattr(new_cls, 'meta_type') == 'market' , f'{name} must be a market factor'
            elif issubclass(new_cls , AffiliateFactorCalculator):
                assert getattr(new_cls, 'meta_type') == 'affiliate' , f'{name} must be a affiliate factor'
            elif issubclass(new_cls , PoolingCalculator):
                assert getattr(new_cls, 'meta_type') == 'pooling' , f'{name} must be a pooling factor'
            else:
                raise ValueError(f'undefined factor type: {name}')

            assert 'category1' not in dct , f'cannot set category1 in {name} , use the corresponding BaseClass in StockFactorCalculator'
            assert 'category0' not in dct , f'cannot set category0 in {name} , use the corresponding BaseClass in StockFactorCalculator'

            category0 , category1 = getattr(new_cls, 'category0' , ''), getattr(new_cls, 'category1' , '')
            if not category1:
                raise AttributeError(f'class {name} category1 is not set')
            if not category0:
                raise AttributeError(f'class {name} category0 is not set')

            CONF.Factor.STOCK.validate_categories(category0 , category1)
            
            setattr(new_cls , 'calc_factor' , _calc_factor_wrapper(getattr(new_cls , 'calc_factor')))
            setattr(new_cls , 'calc_history' , _calc_factor_wrapper(getattr(new_cls , 'calc_history')))

            cls.registry[name] = new_cls
        return new_cls

    def import_definitions(cls):
        if cls.definition_imported:
            return
        for path in sorted(PATH.fac_def.rglob('*.py')):
            module_name = '.'.join(path.relative_to(PATH.main).with_suffix('').parts)
            import_module(module_name)
        cls.definition_imported = True
        
class FactorCalculator(metaclass=_FactorCalculatorMeta):
    """base class of factor calculator"""
    init_date   : int = -1
    final_date  : int = 99991231
    update_step : int = CONF.Factor.UPDATE.step
    category1 : Literal['weighted' , 'nonlinear' , 'style' , 'market_event' , 'quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''
    updatable = True
    preprocess = True
    
    is_pooling = _FactorPropertyBool('is_pooling')
    is_market = _FactorPropertyBool('is_market')

    meta_type = _FactorMetaType()
    db_src    = _FactorDBSrc()
    db_key    = _FactorDBKey()
    category0 = _FactorPropertyStr('category0')
    factor_name = _FactorPropertyStr('factor_name')
    level = _FactorPropertyStr('level')
    file_name = _FactorPropertyStr('file_name')
    factor_string = _FactorPropertyStr('factor_string')

    factor_calendar = _FactorCalendar('calendar')
    update_calendar = _FactorCalendar('update')
    
    min_date = _FactorStoredDates('min')
    max_date = _FactorStoredDates('max')

    UPDATE_MIN_VALID_COUNT_RELAX : int = 20
    UPDATE_MIN_VALID_COUNT_STRICT : int = 100
    UPDATE_RELAX_DATES : list[int] = []

    def __init__(self , *args , **kwargs):
        super().__init__(*args , **kwargs)
        self._date : int | None = None
        self._df : pd.DataFrame | None = None

    def __repr__(self):
        return f'{self.factor_name.capitalize()} Calculator(init_date={self.init_date},category0={self.category0},category1={self.category1},description={self.description})'
    
    def __call__(self , date : int) -> pd.DataFrame:
        """return factor value of a given date , calculate if not exist"""
        df = self.calc_factor(date)
        return df.to_frame().reset_index() if isinstance(df , pd.Series) else df

    @abstractmethod
    def calc_factor(self , date : int) -> pd.Series | pd.DataFrame:
        """calculate factor value , must return a Series or DataFrame that contains at least secid and factor_value / factor_name columns"""

    @abstractmethod
    def calc_history(self , date : int) -> pd.DataFrame:
        """update all factor history calculations, must be implemented for market factor"""
        if self.meta_type != 'market':
            raise NotImplementedError(f'{self.factor_name} is not a market factor')
        raise NotImplementedError(f'{self.factor_name} factor history is not implemented')

    @abstractmethod
    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False , verbose = False) -> bool:
        """store factor data after calculate"""
        raise NotImplementedError(f'{self.factor_name} calc_and_deploy is not implemented')

    @abstractmethod
    def validate_value(self , df : pd.DataFrame , date : int , strict = False) -> pd.DataFrame:
        """validate factor value of a given date"""
        raise NotImplementedError(f'{self.factor_name} validate_value is not implemented')

    def load_factor(self , date : int | None = None) -> pd.DataFrame:
        """load full factor value of a given date"""
        df = DB.load(self.db_src , self.db_key , date , verbose = False)
        return df

    def eval_factor(self , date : int , verbose : bool = False) -> pd.DataFrame:
        """get factor value of a given date , load if exist , calculate if not exist"""
        try:
            df = self.load_factor(date)
        except Exception as e:
            Logger.error(f'{self.factor_name} at {date} error : {e}')
            df = pd.DataFrame()
        if (df.empty or 
            (self.meta_type == 'market' and 'date' not in df.columns) or 
            (self.meta_type == 'stock' and 'secid' not in df.columns)): 
            self.calc_and_deploy(date , overwrite = True)
            df = self._df
            assert df is not None and not df.empty , f'factor {self.factor_name} is not calculated at {date}'
            if verbose: 
                print(f'{self.factor_name} at {date} recalculated')
        return df

    def eval_factor_series(self ,  date : int , verbose : bool = False) -> pd.Series:
        """get factor value of a given date , load if exist , calculate if not exist , return a Series"""
        df = self.eval_factor(date , verbose)
        if 'secid' in df.columns:
            return df.set_index('secid').iloc[:,0]
        elif 'date' in df.columns:
            return df.set_index('date').iloc[:,0]
        else:
            raise ValueError(f'factor {self.factor_name} at {date} has no secid or date column')

    @classmethod
    def Calc(cls , date : int) -> pd.Series | pd.DataFrame:
        """calculate factor value of a given date"""
        return cls().calc_factor(date)
    
    @classmethod
    def Load(cls , date : int | None) -> pd.DataFrame:
        """load factor value of a given date"""
        return cls().load_factor(date)

    @classmethod
    def Loads(cls , dates : np.ndarray | list[int] , normalize = False , 
              fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
        ) -> pd.DataFrame:
        """load factor values of a given date range""" 
        dates = np.intersect1d(dates , cls.stored_dates())
        df = DB.load_multi(cls.db_src , cls.db_key , dates)
        if cls.meta_type == 'stock' and normalize:
            df = StockFactor.normalize_df(df , fill_method = fill_method)
        return df

    @classmethod
    def Eval(cls , date : int) -> pd.DataFrame:
        """get factor value of a given date , load if exist , calculate if not exist"""
        return cls().eval_factor(date)

    @classmethod
    def EvalSeries(cls , date : int) -> pd.Series:
        """get factor value of a given date , load if exist , calculate if not exist , return a Series"""
        return cls().eval_factor_series(date)

    @classmethod
    def FactorDates(cls , start : int | None = 20170101 , end : int | None = None , step : int = 1) -> np.ndarray:
        """get factor dates of a given date range"""
        possible_dates = np.union1d(cls.factor_calendar , cls.stored_dates())
        assert step < cls.update_step or step % cls.update_step == 0 , f'step {step} should be a multiple of or less than {cls.update_step}'
        dates = CALENDAR.slice(possible_dates , start , end)
        dates = dates[dates <= CALENDAR.updated()]
        if step > cls.update_step:
            dates = dates[::int(step/cls.update_step)]
        else:
            dates = np.intersect1d(dates , possible_dates)
        return dates

    @classmethod
    def Factor(cls , dates : np.ndarray | list[int] , normalize = True , 
               fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
               multi_thread = True , ignore_error = True , verbose = False) -> StockFactor:
        """get factor values of a given date range , load if exist , calculate if not exist"""
        if len(dates) == 0: 
            return StockFactor()
        calc = cls()
        func_calls = {date:(calc.eval_factor , {'date' : date , 'verbose' : verbose}) for date in dates}
        dfs = parallel(func_calls , method = multi_thread , ignore_error = ignore_error)

        factor = StockFactor(dfs)
        if normalize: 
            factor.normalize(fill_method , inplace = True)
        return factor

    @classmethod
    def FastTest(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , **kwargs) -> StockFactor:
        """fast test of factor values of a given date range"""
        dates = cls.FactorDates(start , end , step)
        factor = cls.Factor(dates , **kwargs)
        factor.fast_analyze(nday = step)
        return factor

    @classmethod
    def target_path(cls , date : int | None = None) -> Path:
        """full test of factor values of a given date range"""
        return DB.path(cls.db_src , cls.db_key , date)

    @classmethod
    def clear_stored_data(cls , date : int) -> bool:
        """clear stored data of factor"""
        path = cls.target_path(date)
        if not path.exists():
            return False
        else:
            path.unlink(missing_ok = True)
            return True

    @classmethod
    def missing_dates(cls , start : int | None = None , end : int | None = None , overwrite = False) -> np.ndarray:
        """returntarget dates of factor updates"""
        dates = CALENDAR.slice(cls.update_calendar , start , end)
        if not overwrite: 
            dates = CALENDAR.diffs(dates , cls.stored_dates())
        return dates

    @classmethod
    def target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False) -> np.ndarray:
        """returntarget dates of factor updates"""
        return cls.missing_dates(start , end , overwrite)
    
    @classmethod
    def stored_dates(cls) -> np.ndarray:
        """return stored dates of factor data"""
        return DB.dates(cls.db_src , cls.db_key)

    @classmethod
    def get_min_date(cls) -> int:
        """return minimum date of stored factor data"""
        return DB.min_date(cls.db_src , cls.db_key)

    @classmethod
    def get_max_date(cls) -> int:
        """return maximum date of stored factor data"""
        return DB.max_date(cls.db_src , cls.db_key)
    
    @classmethod
    def has_date(cls , date : int) -> bool:
        """check if factor data exists for a given date"""
        return DB.path(cls.db_src , cls.db_key , date).exists()

    @classmethod
    def update_all(cls , start : int | None = None , end : int | None = None , overwrite = False , verbose = False) -> None:
        """update factor data and stats of a given date"""
        cls.update_all_factors(start = start , end = end , overwrite = overwrite , verbose = verbose)
        cls.update_all_stats(start = start , end = end , overwrite = overwrite , verbose = verbose)

    @classmethod
    def update_all_factors(cls , start : int | None = None , end : int | None = None , overwrite = False , verbose = False) -> None:
        """update all factor data until date"""
        dates = cls.target_dates(start = start , end = end , overwrite = overwrite)
        calc = cls()
        for date in dates:
            calc.update_day_factor(date , overwrite = overwrite , show_success = verbose , show_warning = False)

    @classmethod
    def update_all_stats(cls , start : int | None = None , end : int | None = None , overwrite = False , verbose = False) -> None:
        """update all factor stats until date"""
        target_dates = cls.stats_target_dates(start = start , end = end , overwrite = overwrite)
        for stats_type , dates in target_dates.items():
            cls.update_periodic_stats(stats_type , dates , overwrite = overwrite , verbose = verbose)

    def update_day_factor(self , date : int , overwrite = False , show_success = False , show_warning = False ,catch_errors : tuple[type[Exception],...] = ()) -> bool:
        """update factor data of a given date"""
        if show_warning and date not in CONF.Factor.UPDATE.target_dates:
            print(f'Warning: {self.factor_string} at date {date} is not in CONF.Factor.UPDATE.target_dates')
        prefix = f'{self.factor_string} at date {date}'
        try:
            done = self.calc_and_deploy(date , overwrite = overwrite , verbose = show_success)
        except catch_errors as e:
            print(f'{prefix} failed: {e}')
            traceback.print_exc()
            return False
        return done

    @classmethod
    def update_periodic_stats(cls , stats_type : Literal['daily' , 'weekly'] , dates : np.ndarray | list[int] | None , overwrite = False , verbose = False) -> None:
        """update factor daily or weekly stats"""
        if dates is None:
            dates = cls.stats_target_dates()[stats_type]
        elif len(dates) > 0 and not overwrite:
            dates = np.intersect1d(dates , cls.stats_target_dates()[stats_type])
        if len(dates) == 0:
            return
        
        old_df = DB.load(f'factor_stats_{stats_type}' , cls.db_key , verbose = False)
        new_df = getattr(cls.Factor(dates) , f'{stats_type}_stats')()
        df = pd.concat([old_df , new_df]).drop_duplicates(subset = ['date'] , keep = 'last').\
            sort_values('date').reset_index(drop = True)
        DB.save(df , f'factor_stats_{stats_type}' , cls.db_key , verbose = False)
        if verbose:
            print(f'Updated {stats_type} stats of {cls.factor_name} for {len(dates)} dates')

    @classmethod
    def update_daily_stats(cls , dates : np.ndarray | list[int] | None , overwrite = False , verbose = False) -> None:
        """update factor daily stats"""
        cls.update_periodic_stats('daily' , dates , overwrite , verbose)

    @classmethod
    def update_weekly_stats(cls , dates : np.ndarray | list[int] | None , overwrite = False , verbose = False) -> None:
        """update factor weekly stats"""
        cls.update_periodic_stats('weekly' , dates , overwrite , verbose)

    @classmethod
    def daily_stats(cls) -> pd.DataFrame:
        """return stats DataFrame of a given stats type"""
        return DB.load('factor_stats_daily' , cls.db_key , verbose = False)

    @classmethod
    def weekly_stats(cls) -> pd.DataFrame:
        """return stats DataFrame of a given stats type"""
        return DB.load('factor_stats_weekly' , cls.db_key , verbose = False)

    @classmethod
    def stats_stored_dates(cls) -> dict[str , np.ndarray]:
        """return dates of factor stats"""
        stats_types = ['daily' , 'weekly']
        dates : dict[str , np.ndarray] = {}
        for stats_type in stats_types:
            df = DB.load(f'factor_stats_{stats_type}' , cls.db_key , verbose = False)
            if df.empty:
                dates[stats_type] = np.array([] , dtype = int)
            else:
                dates[stats_type] = df['date'].to_numpy(int)
        return dates

    @classmethod
    def stats_target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False) -> dict[Literal['daily' , 'weekly'] , np.ndarray]:
        """return dates of factor stats"""
        factor_stored_dates = CALENDAR.slice(cls.stored_dates() , start , end)
        stats_stored_dates = cls.stats_stored_dates()
        target_dates = {}
        skip_days = {
            'daily' : 1,
            'weekly' : 5,
        }
        for key , dates in stats_stored_dates.items():
            max_date = CALENDAR.td(CALENDAR.updated() , -skip_days[key])
            target = factor_stored_dates[factor_stored_dates <= max_date]
            if not overwrite:
                target = np.setdiff1d(target , dates)
            target_dates[key] = target
        return target_dates

    @classmethod
    def iter_calculators(cls , all = True , selected_factors : list[str] | None = None , **kwargs) -> Generator['FactorCalculator' , None , None]:
        """
        iterate over calculators
        return a list of factor instances with given attributes
        is_pooling : bool | None = None
        is_market : bool | None = None
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        meta_type : Literal['market' , 'stock' , 'affiliate' , 'pooling'] | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        updatable : bool | None = None
        """
        cls.import_definitions()
        for name , calculator in cls.registry.items():
            if ((name in selected_factors) if selected_factors else all) and calculator.match_attrs(**kwargs):
                yield calculator()

    @classmethod
    def match_attrs(cls , **kwargs) -> bool:
        """check if the factor matches the given attributes"""
        kwargs = {k : v for k , v in kwargs.items() if v is not None}
        if len(kwargs) == 0:
            return True
        conditions : list[bool] = []
        for k , v in kwargs.items():
            attr = getattr(cls , k , None)
            if isinstance(v , str): 
                v = v.replace('\\' , '/')
            if isinstance(attr , str):
                attr = attr.replace('\\' , '/')
            conditions.append(attr == v)
        return not conditions or all(conditions)

class StockFactorCalculator(FactorCalculator):
    """base class of factor calculator"""
    def calc_history(self , date : int) -> pd.DataFrame:
        """update all factor history calculations, must be implemented for market factor"""
        raise NotImplementedError(f'{self.factor_name} : fill history should not be implemented for stock factor')

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False , verbose = False) -> bool:
        """store factor data after calculate"""
        if not overwrite and DB.path(self.db_src , self.db_key , date).exists(): 
            return False

        df = self(date)
        df = self.validate_value(df , date , strict = strict_validation)
        
        self._df = df
        self._date = date

        return DB.save(df , self.db_src , self.db_key , date , verbose = verbose)

    def validate_value(self , df : pd.DataFrame , date : int , strict = False) -> pd.DataFrame:
        """
        validate factor value of a given date
        1. valid count should be >= UPDATE_MIN_VALID_COUNT_STRICT or UPDATE_MIN_VALID_COUNT_RELAX
        2. factor_value must not have infinite values
        """

        values = df.loc[:,self.factor_name]
        actual_valid_count = values.notna().sum()

        mininum_valid_count = self.UPDATE_MIN_VALID_COUNT_STRICT
        if date in self.UPDATE_RELAX_DATES or not strict:
            mininum_valid_count = self.UPDATE_MIN_VALID_COUNT_RELAX

        if actual_valid_count < mininum_valid_count:
            raise ValueError(f'{self.factor_name} at {date} must have at least {mininum_valid_count} valid values , but got {actual_valid_count}')
        
        if np.isinf(values).any():
            raise ValueError(f'{self.factor_name} at {date} must not have infinite values , but got {np.isinf(df).sum()} infs')
        return df

class AffiliateFactorCalculator(FactorCalculator):
    """base class of affiliate factor calculator (no need to calculate at all)"""
    load_db_src : str = 'models'
    load_db_key : str = 'tushare_cne5_exp'
    load_col_name : str = ''

    def calc_history(self , date : int) -> pd.DataFrame:
        """no need to validate value for affiliate factor"""
        raise NotImplementedError(f'{self.factor_name} factor history is not implemented')

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False , verbose = False) -> bool:
        """store factor data after calculate"""
        return False

    def validate_value(self , df : pd.DataFrame , *args , **kwargs) -> pd.DataFrame:
        """no need to validate value for affiliate factor"""
        raise NotImplementedError(f'{self.factor_name} validate_value is not implemented')

    @abstractmethod
    def load_factor(self , date : int) -> pd.DataFrame:
        """load full factor value of a given date"""
        df = DB.load(self.load_db_src , self.load_db_key , date).loc[:,['secid' , self.load_col_name]]
        df = df.rename(columns = {self.load_col_name : self.factor_name})
        return df

    def eval_factor(self , date : int , verbose : bool = False) -> pd.DataFrame:
        """get factor value of a given date , load if exist , calculate if not exist"""
        return self.load_factor(date)

    @classmethod
    def Loads(cls , dates : np.ndarray | list[int] , normalize = False , 
              fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
        ) -> pd.DataFrame:
        """load factor values of a given date range"""
        dates = np.intersect1d(dates , cls.stored_dates())
        df = DB.load_multi(cls.load_db_src , cls.load_db_key , dates).loc[:,['secid' , 'date' , cls.load_col_name]]
        df = df.rename(columns = {cls.load_col_name : cls.factor_name})
        return df

    @classmethod
    def Factor(cls , dates : np.ndarray | list[int] , normalize = True , 
               fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
               **kwargs) -> StockFactor:
        """get factor values of a given date range , load if exist , calculate if not exist"""
        if len(dates) == 0: 
            return StockFactor()
    
        factor = StockFactor(cls.Loads(dates))
        if normalize: 
            factor.normalize(fill_method , inplace = True)
        return factor

    @classmethod
    def target_path(cls , date : int | None = None) -> Path:
        """full test of factor values of a given date range"""
        raise NotImplementedError(f'{cls.factor_name} target_path is not implemented')

    @classmethod
    def clear_stored_data(cls , date : int) -> bool:
        """clear stored data of factor"""
        raise NotImplementedError(f'{cls.factor_name} clear_stored_data is not implemented')

    @classmethod
    def missing_dates(cls , start : int | None = None , end : int | None = None , overwrite = False) -> np.ndarray:
        """returntarget dates of factor updates"""
        return np.array([], dtype = int)

    @classmethod
    def target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False) -> np.ndarray:
        """returntarget dates of factor updates"""
        return np.array([], dtype = int)
    
    @classmethod
    def stored_dates(cls) -> np.ndarray:
        """return stored dates of factor data"""
        return CALENDAR.slice(DB.dates(cls.load_db_src , cls.load_db_key) , cls.init_date)

    @classmethod
    def get_min_date(cls) -> int:
        """return minimum date of stored factor data"""
        return max(cls.init_date , DB.min_date(cls.load_db_src , cls.load_db_key))

    @classmethod
    def get_max_date(cls) -> int:
        """return maximum date of stored factor data"""
        return min(cls.final_date , DB.max_date(cls.load_db_src , cls.load_db_key))
    
    @classmethod
    def has_date(cls , date : int) -> bool:
        """check if factor data exists for a given date"""
        return DB.path(cls.load_db_src , cls.load_db_key , date).exists()

    @classmethod
    def update_all_factors(cls , start : int | None = None , end : int | None = None , overwrite = False , verbose = False) -> None:
        """update all factor data until date"""
        return

    def update_day_factor(self , date : int , overwrite = False , show_success = False , show_warning = False ,catch_errors : tuple[type[Exception],...] = ()) -> bool:
        """update factor data of a given date"""
        return True

class MarketFactorCalculator(FactorCalculator):
    """base class of market factor calculator"""
    @classmethod
    def recalculate(cls , date : int | None = None , verbose = True) -> bool:
        """recalculate factor data of a given date"""
        calc_date = CALENDAR.updated() if date is None else int(date)
        instance = cls()
        df = instance.calc_history(calc_date)
        df = instance.validate_value(df , calc_date , strict = True)
        return DB.save(df , cls.db_src , cls.db_key , verbose = verbose)

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False , verbose = False) -> bool:
        """store factor data after calculate"""
        if not DB.path(self.db_src , self.db_key).exists():
            return self.recalculate(date , verbose = verbose)
        
        old_df = DB.load(self.db_src , self.db_key , verbose = False)
        if not overwrite and not old_df.empty and any(old_df['date'] == date):
            return False
        df = self(date)
        df = self.validate_value(df , date , strict = strict_validation)
        df = pd.concat([old_df , df]).drop_duplicates(subset = ['date'] , keep = 'first').\
            sort_values('date').reset_index(drop = True)

        return DB.save(df , self.db_src , self.db_key , verbose = verbose)

    def validate_value(self , df : pd.DataFrame , date : int , strict = False) -> pd.DataFrame:
        """
        validate market factor value of a given date
        1. factor_value must not have not finite values
        """
        values = df.loc[:,self.factor_name]
        if not np.isfinite(values).all():
            raise ValueError(f'factor_value must not have not finite values , but got {np.isfinite(values).sum()} not finite values')
        return df

    @classmethod
    def update_daily_stats(cls , dates : np.ndarray | list[int] | None , overwrite = False) -> None:
        """update factor daily stats is not implemented for market factor"""
        return None

    @classmethod
    def update_weekly_stats(cls , dates : np.ndarray | list[int] | None , overwrite = False) -> None:
        """update factor weekly stats is not implemented for market factor"""
        return None

    @classmethod
    def stats_target_dates(cls , *args , **kwargs) -> dict[str , np.ndarray]:
        """return target dates of factor stats"""
        return {
            'daily' : np.array([] , dtype = int),
            'weekly' : np.array([] , dtype = int),
        }

    @classmethod
    def Loads(cls , start : int | None = None , end : int | None = None , fillna = False , fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'indus_median') -> pd.DataFrame:
        """load factor values of a given date range"""
        df = DB.load(cls.db_src , cls.db_key , verbose = False)
        if start is not None or end is not None:
            df = df.loc[df['date'].isin(CALENDAR.td_within(start , end))].reset_index(drop = True).copy()
        return df

    @classmethod
    def Factor(cls , *args , **kwargs) -> StockFactor:
        """get factor values of a given date range , load if exist , calculate if not exist"""
        raise NotImplementedError(f'{cls.factor_name} : Factor should not be implemented for market factor')

    @classmethod
    def target_dates(cls , start : int | None = None , end : int | None = None , overwrite = False) -> np.ndarray:
        """returntarget dates of factor updates"""
        return cls.missing_dates(start , end , overwrite)[-1:]
    
    @classmethod
    def stored_dates(cls) -> np.ndarray:
        """return stored dates of factor data"""
        df = DB.load(cls.db_src , cls.db_key , verbose = False)
        dates = df['date'].to_numpy(int) if not df.empty else np.array([])
        return dates

    @classmethod
    def get_min_date(cls) -> int:
        """return minimum date of stored factor data"""
        dates = cls.stored_dates()
        return dates.min() if len(dates) > 0 else 99991231

    @classmethod
    def get_max_date(cls) -> int:
        """return maximum date of stored factor data"""
        dates = cls.stored_dates()
        return dates.max() if len(dates) > 0 else 0
    
    @classmethod
    def has_date(cls , date : int) -> bool:
        """check if factor data exists for a given date"""
        return any(cls.stored_dates() == date)

    @classmethod
    def clear_stored_data(cls , date : int) -> bool:
        """clear stored data of factor"""
        path = cls.target_path(date)
        if not path.exists():
            return False
        else:
            df = DB.load_df(path)
            if df.empty:
                return False
            df = df.loc[df['date'] <= date]
            DB.save_df(df , path , overwrite = True)
            return True

class PoolingCalculator(FactorCalculator):
    """base class of factor calculator"""
    def calc_history(self , date : int) -> pd.DataFrame:
        """update all factor history calculations, must be implemented for market factor"""
        raise NotImplementedError(f'{self.factor_name} : fill history should not be implemented for stock factor')

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False , verbose = False) -> bool:
        """store factor data after calculate"""
        if not overwrite and DB.path(self.db_src , self.db_key , date).exists(): 
            return False

        df = self(date)
        df = self.validate_value(df , date , strict = strict_validation)
        
        self._df = df
        self._date = date

        return DB.save(df , self.db_src , self.db_key , date , verbose = verbose)

    def validate_value(self , df : pd.DataFrame , date : int , strict = False) -> pd.DataFrame:
        """
        validate factor value of a given date
        1. valid count should be >= UPDATE_MIN_VALID_COUNT_STRICT or UPDATE_MIN_VALID_COUNT_RELAX
        2. factor_value must not have infinite values
        """

        values = df.loc[:,self.factor_name]
        actual_valid_count = values.notna().sum()

        mininum_valid_count = self.UPDATE_MIN_VALID_COUNT_STRICT
        if date in self.UPDATE_RELAX_DATES or not strict:
            mininum_valid_count = self.UPDATE_MIN_VALID_COUNT_RELAX

        if actual_valid_count < mininum_valid_count:
            raise ValueError(f'{self.factor_name} at {date} must have at least {mininum_valid_count} valid values , but got {actual_valid_count}')
        
        if np.isinf(values).any():
            raise ValueError(f'{self.factor_name} at {date} must not have infinite values , but got {np.isinf(df).sum()} infs')
        return df

class QualityFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: fundamental , category1: quality"""
    category1 = 'quality'

class GrowthFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: fundamental , category1: growth"""
    category1 = 'growth'

class ValueFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: fundamental , category1: value"""
    category1 = 'value'

class EarningFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: fundamental , category1: earning"""
    category1 = 'earning'

class SurpriseFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: analyst , category1: surprise"""
    category1 = 'surprise'

class CoverageFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: analyst , category1: coverage"""
    category1 = 'coverage'

class ForecastFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: analyst , category1: forecast"""
    category1 = 'forecast'

class AdjustmentFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: analyst , category1: adjustment"""
    category1 = 'adjustment'

class HfMomentumFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: high_frequency , category1: hf_momentum"""
    category1 = 'hf_momentum'

class HfVolatilityFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: high_frequency , category1: hf_volatility"""
    category1 = 'hf_volatility'

class HfCorrelationFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: high_frequency , category1: hf_correlation"""
    category1 = 'hf_correlation'

class HfLiquidityFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: high_frequency , category1: hf_liquidity"""
    category1 = 'hf_liquidity'

class MomentumFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: behavior , category1: momentum"""
    category1 = 'momentum'

class VolatilityFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: behavior , category1: volatility"""
    category1 = 'volatility'

class CorrelationFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: behavior , category1: correlation"""
    category1 = 'correlation'

class LiquidityFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: behavior , category1: liquidity"""
    category1 = 'liquidity'

class HoldingFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: money_flow , category1: holding"""
    category1 = 'holding'

class TradingFactor(StockFactorCalculator):
    """Factor Calculator of meta_type: stock , category0: money_flow , category1: trading"""
    category1 = 'trading'

class StyleFactor(AffiliateFactorCalculator):
    """Factor Calculator of meta_type: affiliate , category0: risk , category1: style"""
    init_date = 20110101
    category1 = 'style'
    load_db_src = 'models'
    load_db_key = 'tushare_cne5_exp'
    update_step = 1
    preprocess = False

class SellsideFactor(AffiliateFactorCalculator):
    """Factor Calculator of meta_type: affiliate , category0: external , category1: sellside"""
    category1 = 'sellside'
    load_db_src = 'sellside'
    update_step = 1
    preprocess = False

class MarketEventFactor(MarketFactorCalculator):
    """Factor Calculator of meta_type: market , category0: market , category1: market_event"""
    category1 = 'market_event'

class WeightedPoolingCalculator(PoolingCalculator):
    """Factor Calculator of meta_type: pooling , category0: pooling , category1: weighted"""
    category1 = 'weighted'

    def load_pooling_weight(self) -> pd.DataFrame:
        """load pooling weight of a given date"""
        df = DB.load('pooling_weight' , self.db_key , verbose = False)
        return df

    def purge_pooling_weight(self , confirm : bool = False) -> None:
        """purge pooling weight of a given date"""
        if confirm:
            DB.path('pooling_weight' , self.db_key).unlink(missing_ok = True)

    def get_pooling_weight(self , date : int) -> pd.DataFrame:
        """get pooling weight of a given date"""
        if not hasattr(self , '_loaded_weight'):
            self._loaded_weight = self.load_pooling_weight()
        if self._loaded_weight.empty or date not in self._loaded_weight['date']:
            self.update_all_pooling_weight(date = date)
            self._loaded_weight = self.load_pooling_weight()
        if self._loaded_weight.empty:
            return pd.DataFrame()
        weight = self._loaded_weight.query('date == @date').copy().set_index('date')
        if weight.empty:
            print(self._loaded_weight['date'].unique())
            raise ValueError(f'pooling weight is empty for {date}')
        return weight

    @abstractmethod
    def calc_pooling_weight(self , start : int | None = None , end : int | None = None , dates : np.ndarray | None = None , overwrite = False , verbose = False) -> pd.DataFrame:
        """calculate pooling weight of a given date range"""
        raise NotImplementedError(f'{self.factor_name} : calc_pooling_weight should not be implemented for weighted pooling')

    @classmethod
    def update_all(cls , start : int | None = None , end : int | None = None , overwrite = False , verbose = False) -> None:
        """update factor data and stats of a given date"""
        cls.update_all_pooling_weight(date = end , verbose = verbose)
        cls.update_all_factors(start = start , end = end , overwrite = overwrite , verbose = verbose)
        cls.update_all_stats(start = start , end = end , overwrite = overwrite , verbose = verbose)

    @classmethod
    def update_all_pooling_weight(cls , date : int | None = None , overwrite = False , verbose = False) -> None:
        """update all factor data until date"""
        calc = cls()
        target_dates = CALENDAR.slice(calc.update_calendar , 0 , date)
        if not overwrite:
            old_weights = calc.load_pooling_weight()
            if not old_weights.empty:
                target_dates = np.setdiff1d(target_dates , old_weights['date'])
        else:
            old_weights = pd.DataFrame()
        if len(target_dates) == 0:
            return
        new_weight = calc.calc_pooling_weight(dates = target_dates)
        if 'date' not in new_weight.columns:
            new_weight = new_weight.reset_index(drop = False)
        weights = pd.concat([old_weights , new_weight]).drop_duplicates(subset = ['date'] , keep = 'last').sort_values('date')
        DB.save(weights , 'pooling_weight' , calc.db_key , verbose = verbose)

class NonlinearPoolingCalculator(PoolingCalculator):
    """Factor Calculator of meta_type: pooling , category0: pooling , category1: nonlinear"""
    category1 = 'nonlinear'
