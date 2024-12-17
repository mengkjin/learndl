import numpy as np
import pandas as pd
import importlib.util
import inspect 

from abc import abstractmethod
from dataclasses import dataclass
from itertools import combinations
from typing import Any , Literal , Type , ClassVar

from src.basic import PATH , CALENDAR , IS_SERVER
from src.data import DATAVENDOR
from src.func.singleton import SingletonABCMeta , singleton
from src.func.classproperty import classproperty_str
from src.func.parallel import parallel
from src.factor.util import StockFactor

class DateError(Exception): ...
class CategoryError(Exception): ...

UPDATE_START = 20070101 if IS_SERVER else 20241101
UPDATE_END   = 20991231 if IS_SERVER else 20241231
@dataclass  
class FactorUpdateJob:
    obj : 'StockFactorCalculator'
    date : int

    def __post_init__(self):
        self.done = False
        if self.date < UPDATE_START or self.date > UPDATE_END: 
            raise DateError(f'date is should be between {UPDATE_START} and {UPDATE_END}, but got {self.date}')
    @property
    def level(self): return self.obj.level
    @property
    def factor_name(self): return self.obj.factor_name
    @property
    def sort_key(self): return (self.level , self.date , self.factor_name)
    def do(self , overwrite = False , show_progress = True , catch_errors : tuple[Type[Exception],...] = ()):
        try:
            self.obj.calc_and_deploy(self.date , overwrite = overwrite , show_progress = show_progress)
            self.done = True
        except catch_errors as e:
            print(f'Factor : {self.factor_name} update at date {self.date} failed: {e}')
    
@singleton
class FactorUpdateJobManager:
    CATCH_ERRORS = (ValueError , TypeError)
    def __init__(self , multi_thread = True):
        self.jobs : list[FactorUpdateJob] = []
        self.parallel_type : Literal['thread' , 'process' , False] | None = 'thread' if multi_thread else None
    def __repr__(self):
        return f'FactorUpdateJobs({len(self.jobs)} jobs)'
    def to_dataframe(self):
        columns = ['level' , 'date' , 'factor']
        return pd.DataFrame([(job.level , job.date , job.factor_name) for job in self.jobs] , columns=columns).sort_values(by=columns)
    def filter(self , jobs : list[FactorUpdateJob] , level : str , date : int):
        return [job for job in jobs if job.level == level and job.date == date]
    def clear(self): self.jobs.clear()
    def sort(self): self.jobs.sort(key=lambda x: x.sort_key)
    def append(self , job : FactorUpdateJob): self.jobs.append(job)
    def proceed(self , overwrite = False , show_progress = True , ignore_error = False , max_update_groups : int | None = None):
        '''perform all update jobs , if factor_name is not None , only perform update jobs of the factor'''
        groups = sorted(set((job.level , job.date) for job in self.jobs))[:max_update_groups]
        def do_job(job : FactorUpdateJob): job.do(overwrite , show_progress , self.CATCH_ERRORS if ignore_error else ())
        for level , date in groups:
            DATAVENDOR.data_storage_control()
            jobs = self.filter(self.jobs , level , date)
            parallel(do_job , jobs , type = self.parallel_type)
            [self.jobs.remove(job) for job in jobs if job.done]

UPDATE_JOBS = FactorUpdateJobManager()

class StockFactorCalculator(metaclass=SingletonABCMeta):
    init_date : int = -1
    category0 : Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative'] | Any
    category1 : Literal['quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''

    INIT_DATE = 20070101
    CATEGORY0_SET = ['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative']
    CATEGORY1_SET = {
        'fundamental' : ['quality' , 'growth' , 'value' , 'earning'] ,
        'analyst' : ['surprise' , 'coverage' , 'forecast' , 'adjustment'] ,
        'high_frequency' : ['hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity'] ,
        'behavior' : ['momentum' , 'volatility' , 'correlation' , 'liquidity'] ,
        'money_flow' : ['holding' , 'trading'] ,
        'alternative' : None
    }

    def __new__(cls , *args , **kwargs):
        return super().__new__(cls.validate_attrs())
    
    @classmethod
    def Calc(cls , date : int):
        return cls().calc_factor(date)
    
    @classmethod
    def Load(cls , date : int):
        return cls().load_factor(date)

    @classmethod
    def Eval(cls , date : int):
        return cls().eval_factor(date)

    @classmethod
    def Factor(cls , start : int | None = 20170101 , end : int | None = None , step : int = 10 , normalize = True , 
               fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
               weighted_whiten = False , order = ['fillna' , 'whiten' , 'winsor'] ,
               multi_thread = True , ignore_error = True):
        dates = CALENDAR.slice(CALENDAR.td_within(start , end , step) , cls.init_date , CALENDAR.updated())
        calc = cls()
        def calculate_factor(date):
            df = calc.eval_factor(date)
            print(f'{calc.factor_name} at {date} calculated')
            return df
        dfs = parallel(calculate_factor , dates , dates , type = 'thread' if multi_thread else None , ignore_error = ignore_error)
        factor = StockFactor(dfs)
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

    def eval_factor(self , date : int):
        df = self.load_factor(date)
        return self.calc_factor(date) if df.empty else df
    
    @classmethod
    def calc_factor_wrapper(cls , raw_calc_factor):
        '''validate calculated factor value'''
        def new_calc_factor(self , date):
            date = int(date)
            if date < cls.init_date: 
                raise DateError(f'date should be >= {cls.init_date} for Factor {cls.factor_name} , but got {date}')
            
            df = raw_calc_factor(self , date)

            if not isinstance(df , pd.Series):  
                raise TypeError(f'calc_factor must return a Series , but got {type(df)} for factor {cls.factor_name}')
            
            if df.empty: df = pd.Series()
            else: df = df.rename(cls.factor_name).replace([np.inf , -np.inf] , np.nan).reindex(DATAVENDOR.secid(date))

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
        return cls.__module__.split('.')[-2]
    
    @classproperty_str
    def file_name(cls) -> str:
        '''file name of the factor'''
        return cls.__module__.split('.')[-1]

    def __repr__(self):
        return f'{self.factor_name}(from{self.init_date},{self.category0},{self.category1})'
    
    def __call__(self , date : int):
        '''
        return factor value of a given date , calculate if not exist
        '''
        return self.calc_factor(date)

    @classmethod
    def stored_dates(cls): 
        '''return list of stored dates of factor data'''
        return PATH.factor_dates(cls.factor_name)
    
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
            raise DateError(f'init_date should be later than {cls.INIT_DATE} , but got {cls.init_date}')

        if cls.category0 not in cls.CATEGORY0_SET:
            raise CategoryError(f'category0 is should be in {cls.CATEGORY0_SET}, but got {cls.category0}')
        
        if not cls.category1:
            raise CategoryError('category1 is not set')
        
        if (category1_list := cls.CATEGORY1_SET[cls.category0]):
            if cls.category1 not in category1_list:
                raise CategoryError(f'category1 is should be in {category1_list}, but got {cls.category1}')

        if not cls.description:
            raise CategoryError('description is not set')
        
        return cls
        
    @classmethod
    def validate_value(cls , df : pd.DataFrame , strict = False):
        '''validate factor value'''
        mininum_finite_count = 100 if strict else 0
        actual_finite_count = np.isfinite(df[cls.factor_name].to_numpy()).sum()

        if actual_finite_count < mininum_finite_count:
            raise ValueError(f'factor_value must have at least {mininum_finite_count} finite values , but got {actual_finite_count}')
        return df

    def calc_and_deploy(self , date : int , strict_validation = True , overwrite = False , show_progress = False):
        '''store factor data after calculate'''
        if not overwrite and PATH.factor_path(self.factor_name , date).exists(): return self
        df = self.calc_factor(date).rename(self.factor_name).to_frame()
        df = self.validate_value(df , strict = strict_validation)
        saved = PATH.factor_save(df , self.factor_name , date , overwrite)
        if show_progress:
            factor_str = f'Factor {self.level}/{self.category0}/{self.category1}/{self.factor_name}'
            if saved: print(f'{factor_str} at date {date} deploy successful')
            else: print(f'{factor_str} at date {date} already there')
        return self

    @classmethod
    def update_jobs(cls , start : int | None = None , end : int | None = None , overwrite = False):
        dates = CALENDAR.td_within(max(cls.init_date , UPDATE_START) , min(CALENDAR.updated() , UPDATE_END))
        dates = CALENDAR.slice(dates , start , end)
        if not overwrite: dates = CALENDAR.diffs(dates , cls.stored_dates())
        self = cls()
        [UPDATE_JOBS.append(FactorUpdateJob(self , d)) for d in dates]
    
class StockFactorHierarchy:
    '''hierarchy of factor classes'''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.definition_path =  PATH.main.joinpath('src_factor_definition')
        assert self.definition_path.exists() , f'{self.definition_path} does not exist'
        self.load()

    def __repr__(self):
        str_level_factors = [','.join(f'{level}({len(factors)})' for level , factors in self.hier.items())]
        return f'StockFactorHierarchy({str_level_factors})'
    
    def __iter__(self):
        '''return a generator of factor classes'''
        return (cls for level in self.iter_levels() for cls in self.iter_level_factors(level))
    
    def __getitem__(self , key : str):
        '''return a list of factor classes in a given level / or a factor class by factor_name'''
        return self.pool[key] if key in self.pool else self.hier[key]
    
    @staticmethod
    def factor_filter(stock_factor_cls : Type[StockFactorCalculator] , **kwargs):
        '''filter factor by given attributes'''
        conditions : list[bool] = []
        for k , v in kwargs.items():
            if v is None: continue
            attr = getattr(stock_factor_cls , k)
            if isinstance(v , str): 
                v = v.replace('\\' , '/')
                attr = attr.replace('\\' , '/')
            conditions.append(attr == v)
        return not conditions or all(conditions)

    def load(self):     
        '''load all factor classes from definition path'''
        self.pool : dict[str , Type[StockFactorCalculator]] = {}   
        self.hier : dict[str , list[Type[StockFactorCalculator]]] = {}
        for level_path in self.definition_path.iterdir():
            if not level_path.is_dir(): continue
            if not level_path.name.startswith('level'): continue

            for file_path in level_path.rglob('*.py'):
                level_name = level_path.stem
                file_name = str(file_path.relative_to(level_path).with_suffix(''))
                spec_name = f'{level_name}.{file_name}'
                
                spec = importlib.util.spec_from_file_location(spec_name, file_path)
                assert spec is not None and spec.loader is not None , f'{file_path} is not a valid module'
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for _ , obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == spec_name and issubclass(obj , StockFactorCalculator) and (obj is not StockFactorCalculator):
                        assert obj.__name__ not in self.pool , f'{obj.__name__} in module {spec_name} is duplicated'                        
                        self.pool[obj.__name__] = obj
                        if level_path.stem not in self.hier: self.hier[level_name] = []
                        self.hier[level_name].append(obj)

        return self

    def factor_df(self , **kwargs):
        '''
        return a DataFrame of all factors with given attributes
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        attr_list = ['level' , 'file_name' , 'factor_name' , 'init_date' , 'category0' , 'category1' , 'description']
        df_dict = [[getattr(cls , a) for a in attr_list] for cls in self if self.factor_filter(cls , **kwargs)]
        df = pd.DataFrame(df_dict, columns=attr_list)
        return df
    
    def jobs(self):
        '''return a DataFrame of update jobs'''
        return UPDATE_JOBS.to_dataframe()

    def factor_names(self):
        '''return a list of factor names'''
        return [f'{cls.factor_name}({cls.level}.{cls.file_name})' for cls in self]

    def iter_levels(self):
        '''return a list of levels'''
        return iter(self.hier)
    
    def iter_level_factors(self , level : str):
        '''return a list of factor classes in a given level'''
        return (cls for cls in self.hier[level])

    def iter_instance(self , **kwargs):
        '''
        return a list of factor instances with given attributes
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        return (cls() for cls in self if self.factor_filter(cls , **kwargs))
    
    def get_factor(self , factor_name : str):
        '''
        return a factor class by factor_name
        e.g.
        factor_name = 'turn_12m'
        factor_cls = StockFactorHierarchy()[factor_name]
        '''
        return self.pool[factor_name]
    
    def test_calc_all_factors(self , date : int = 20241031 , check_variation = True , check_duplicates = True , 
                              parallel_type : Literal['thread' , 'process' , False] | None = 'thread' , ignore_error = True , verbose = True , **kwargs):
        '''
        test calculation of all factors , if check_duplicates is True , check factors diffs' standard deviation and correlation
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        
        factor_values : dict[str , pd.Series] = {}

        def calculate_factor(obj : StockFactorCalculator):
            factor_value = obj.calc_factor(date)
            valid_ratio = factor_value.dropna().count() / len(factor_value)
            if verbose or valid_ratio < 0.3: 
                print(f'{obj.factor_name} calculated , valid_ratio is {valid_ratio :.2%}')
            return factor_value

        factor_names = [obj.factor_name for obj in self.iter_instance(**kwargs)]
        factor_values : dict[str , pd.Series] = \
            parallel(calculate_factor , self.iter_instance(**kwargs) , factor_names , type = parallel_type , ignore_error = ignore_error)
        self.calc_factor_values = factor_values

        if check_variation:
            abnormal_vars = {}

            for fn in factor_values.keys():
                std = factor_values[fn].std()
                box = factor_values[fn].quantile([0.01 , 0.99]).diff().dropna().astype(float).item()
                
                if std <= 1e-4 or abs(box) <= 1e-4: 
                    abnormal_vars[fn] = {'std':std , 'box':box}
            if len(abnormal_vars) == 0: 
                print('no abnormal factor variation')
            else:
                print(f'abnormal factor variation: {abnormal_vars}')

        if check_duplicates and len(factor_values) <= 100:
            abnormal_diffs = {}
            for fn1 , fn2 in combinations(factor_values.keys() , 2):
                f1 = (factor_values[fn1] - factor_values[fn1].mean()) / factor_values[fn1].std()
                f2 = (factor_values[fn2] - factor_values[fn2].mean()) / factor_values[fn2].std()

                diff = (f1 - f2).fillna(0).abs().std()
                corr = f1.corr(f2)
                if diff <= 0.01 or abs(corr) >= 0.999: 
                    abnormal_diffs[f'{fn1}.{fn2}'] = {'diff_std':diff , 'corr' : corr}
            if len(abnormal_diffs) == 0: 
                print('no abnormal factor diffs')
            else:
                print(f'abnormal factor diffs: {abnormal_diffs}')
        return factor_values
    
    @classmethod
    def update_jobs(cls , start : int | None = None , end : int | None = None , all_factors = False ,overwrite = False , **kwargs):
        '''
        update update jobs for all factors between start and end date
        **kwargs:
            factor_name : str | None = None
            level : str | None = None 
            file_name : str | None = None
            category0 : str | None = None 
            category1 : str | None = None 
        '''
        self = cls()
        UPDATE_JOBS.clear()
        if all_factors:
            [obj.update_jobs(start , end , overwrite) for obj in self.iter_instance(**kwargs)]
        elif kwargs:
            [obj.update_jobs(start , end , overwrite) for obj in self.iter_instance(**kwargs)]
        return self
    
    @classmethod
    def update(cls , show_progress = True , ignore_error = True):
        '''update factor data according to update jobs'''
        cls.update_jobs(all_factors = True , overwrite = False)
        UPDATE_JOBS.proceed(False , show_progress , ignore_error , max_update_groups = 100)

    
