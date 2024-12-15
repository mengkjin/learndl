import numpy as np
import pandas as pd
import importlib.util
import inspect 
import concurrent.futures

from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import combinations
from typing import Any , Literal , Type , final

from src.basic import PATH , CALENDAR , IS_SERVER
from src.data import DATAVENDOR
from src.func.singleton import SingletonABCMeta
from src.func.classproperty import classproperty_str

_FACTOR_UPDATE_JOBS : list[tuple['StockFactorCalculator' , int]] = []

_FACTOR_INIT_DATE = 20070101
_FACTOR_CATEGORY0_SET = ['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative']
_FACTOR_CATEGORY1_SET = {
    'fundamental' : ['quality' , 'growth' , 'value' , 'earning'] ,
    'analyst' : ['surprise' , 'coverage' , 'forecast' , 'adjustment'] ,
    'high_frequency' : ['hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity'] ,
    'behavior' : ['momentum' , 'volatility' , 'correlation' , 'liquidity'] ,
    'money_flow' : ['holding' , 'trading'] ,
    'alternative' : None
}

_FACTOR_UPDATE_START = 20070101 if IS_SERVER else 20241101
_FACTOR_UPDATE_END   = 20991231 if IS_SERVER else 20241231

def insert_update_job(obj : 'StockFactorCalculator' , date : int):
    '''insert a update job to _FACTOR_UPDATE_JOBS'''
    assert date >= _FACTOR_UPDATE_START , f'date is should be greater than or equal to {_FACTOR_UPDATE_START}, but got {date}'
    assert date <= _FACTOR_UPDATE_END   , f'date is should be less than or equal to {_FACTOR_UPDATE_END}, but got {date}'
    _FACTOR_UPDATE_JOBS.append((obj , date))

def perform_update_jobs(overwrite = False , show_progress = True , ignore_error = False , factor_name : str | None = None):
    '''perform all update jobs , if factor_name is not None , only perform update jobs of the factor'''
    _FACTOR_UPDATE_JOBS.sort(key=lambda x: (x[0].level , x[1] , x[0].factor_name))
    date = -1
    for item in _FACTOR_UPDATE_JOBS[:]:
        if item[1] != date: DATAVENDOR.data_storage_control()
        obj , date = item
        if factor_name is not None and obj.factor_name != factor_name: continue
        
        try:
            obj.calculate(date).deploy(overwrite = overwrite , show_progress = show_progress)
            _FACTOR_UPDATE_JOBS.remove(item)
        except Exception as e:
            if ignore_error:
                print(f'Factor : {obj.factor_name} update at date {date} failed: {e}')
            else:
                raise e

class StockFactorCalculator(metaclass=SingletonABCMeta):
    init_date : int = -1
    category0 : Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative'] | Any
    category1 : Literal['quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''

    def __new__(cls , *args , **kwargs):
        return super().__new__(cls.validate_attr())
    
    @classmethod
    def Calc(cls , date : int):
        return cls().calc_factor(date)

    @classmethod
    def Eval(cls , date : int):
        df = PATH.factor_load(cls.factor_name , date , verbose = False)
        if df.empty: return cls.Calc(date)
        else: return df.iloc[:,0]

    @abstractmethod
    def calc_factor(self , date : int) -> pd.Series:
        '''calculate factor value , must have secid and factor_value / factor_name columns'''
    
    @classmethod
    def calc_factor_wrapper(cls , raw_calc_factor):
        '''validate calculated factor value'''
        def new_calc_factor(self , date):
            date = int(date)
            assert date >= cls.init_date , f'for factor {cls.factor_name} , date is should be greater than or equal to {cls.init_date}, but got {date}'
            df = raw_calc_factor(self , date)
            assert isinstance(df , pd.Series) , \
                f'for factor {cls.factor_name} , calc_factor must return a Series , but got {type(df)}'
            df = df.rename(cls.factor_name).replace([np.inf , -np.inf] , np.nan).reindex(DATAVENDOR.secid(date))
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
    
    @classmethod
    def validate_attr(cls):
        '''
        validate attribute of factor
        init_date : must be greater than _FACTOR_INIT_DATE(20070101)
        category0 : must be in _FACTOR_CATEGORY0_SET([fundamental , analyst , high_frequency , behavior , money_flow , alternative])
        category1 : must be in _FACTOR_CATEGORY1_SET[category0] if category1_list is not None , otherwise must be not None
            fundamental : quality , growth , value , earning
            analyst : surprise , coverage , forecast , adjustment
            high_frequency : hf_momentum , hf_volatility , hf_correlation , hf_liquidity
            behavior : momentum , volatility , correlation , liquidity
            money_flow : holding , trading
            alternative : None
        description : must be a non-empty string
        '''
        assert cls.init_date >= _FACTOR_INIT_DATE , f'init_date should be later than {_FACTOR_INIT_DATE} , but got {cls.init_date}'

        assert cls.category0 in _FACTOR_CATEGORY0_SET , \
            f'category0 is should be in {_FACTOR_CATEGORY0_SET}, but got {cls.category0}'
        
        category1_list = _FACTOR_CATEGORY1_SET[cls.category0]
        
        if category1_list is not None:
            assert cls.category1 in category1_list , \
                f'category1 is should be in {category1_list}, but got {cls.category1}'
        else:
            assert cls.category1 , 'category1 is not set'
        
        assert cls.description , 'description is not set'
        return cls

    @final
    def __init__(self):
        if not hasattr(self , 'factors'):
            self.factors : dict[int , pd.DataFrame] = {}

    def __repr__(self):
        return f'{self.factor_name}(from{self.init_date},{self.category0},{self.category1})[{len(self.factors)}dates]'
    
    def __getitem__(self , date : int):
        '''
        return calculated factor value of a given date
        '''
        return self.factors[date]
    
    def __call__(self , date : int):
        '''
        return factor value of a given date , calculate if not exist
        '''
        if date not in self.factors: self.calculate(date)
        return self.factors[date]

    @property
    def stored_dates(self): 
        '''
        return list of stored dates of factor data
        '''
        paths = PATH.list_files(PATH.factor.joinpath(self.factor_name) , recur=True)
        dates = np.array(sorted(PATH.file_dates(paths)) , dtype=int)
        return dates
    
    def factor_values(self):
        '''
        return a DataFrame of all calculated factor values with date and secid index
        '''
        return pd.concat([df.assign(date = d) for d , df in self.factors.items()]).reset_index().set_index(['date','secid'])
    
    def validate_value(self , date : int , df : pd.DataFrame , strict = False):
        '''validate factor value'''

        assert 20991231 >= date >= self.init_date , \
            f'calc_date is should be in [{self.init_date} , 20991231], but got {date}'

        mininum_finite_count = 100 if strict else 0
        actual_finite_count = np.isfinite(df[self.factor_name].to_numpy()).sum()
        if actual_finite_count < mininum_finite_count:
            raise ValueError(f'factor_value must have at least {mininum_finite_count} finite values , but got {actual_finite_count}')
        
        return self

    def calculate(self , date : int | Iterable | Any):
        '''calculate factor value of a given date and store to factor_data'''

        if isinstance(date , Iterable):
            for d in date: self.calculate(d)
        else:
            #date = int(date)
            #assert date >= self.init_date , f'date is should be greater than or equal to {self.init_date}, but got {date}'
            df = self.calc_factor(date)
            self.factors[date] = df.rename(self.factor_name).to_frame()
        return self

    def deploy(self , strict = True , overwrite = False , show_progress = False):
        '''store factor data after calculate'''
        dates = list(self.factors.keys())
        for date in dates:
            df = self.factors.pop(date)
            try:
                self.validate_value(date , df , strict = strict)
                saved = PATH.factor_save(df , self.factor_name , date , overwrite)
                if show_progress:
                    if saved: print(f'Factor : {self.factor_name} at date {date} deploy successful')
                    else: print(f'Factor : {self.factor_name} at date {date} already there')
            except ValueError as e:
                print(f'Factor : {self.factor_name} at date {date} is invalid: {e}')

        return self

    def load(self, date : int | Iterable | Any , factor_name : str | None = None , overwrite = False):
        '''load factor data from storage'''
        if factor_name is None: factor_name = self.factor_name
        if not isinstance(date , Iterable): date = [date]
        for d in date:
            if overwrite or int(d) not in self.factors:
                self.factors[int(d)] = PATH.factor_load(factor_name , date)

    def update_jobs(self , start : int | None = None , end : int | None = None , overwrite = False):
        dates = CALENDAR.td_within(self.init_date , CALENDAR.update_to())
        dates = CALENDAR.slice(dates , _FACTOR_UPDATE_START , _FACTOR_UPDATE_END)
        dates = CALENDAR.slice(dates , start , end)
        if not overwrite: dates = CALENDAR.diffs(dates , self.stored_dates)
        [insert_update_job(self , d) for d in dates]
        return self

    @classmethod
    def Update(cls , overwrite = False , show_progress = True , ignore_error = False):
        '''update factor data from self.init_date to today'''
        perform_update_jobs(overwrite , show_progress , ignore_error , cls.__name__)
    
class StockFactorHierarchy:
    '''
    hierarchy of factor classes
    '''
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
        '''
        return a generator of factor classes
        '''
        return (cls for level in self.iter_levels() for cls in self.iter_level_factors(level))
    
    def __getitem__(self , key : str):
        '''
        return a list of factor classes in a given level / or a factor class by factor_name
        '''
        if key in self.pool: 
            return self.pool[key]
        else:
            return self.hier[key]

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
        return pd.DataFrame([(x.level , d , x) for x,d in _FACTOR_UPDATE_JOBS] , columns=['level' , 'date' , 'factor'])

    def clear_jobs(self):
        '''clear update jobs'''
        _FACTOR_UPDATE_JOBS.clear()
        return self

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
                              multi_thread = True , ignore_error = True , **kwargs):
        '''
        test calculation of all factors , if check_duplicates is True , check factors diffs' standard deviation and correlation
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        
        factor_values : dict[str , pd.Series] = {}

        '''
        for obj in self.iter_instance(**kwargs):
            df = obj.calculate(date).factors[date]
            factor_values[obj.factor_name] = df[obj.factor_name] if isinstance(df , pd.DataFrame) else df
            print(f'{obj.factor_name} calculated , valid_num is {df.dropna().count().item()}')
        
        '''
        def calculate_factor(obj : StockFactorCalculator , date : int):
            factor_value = obj.calculate(date).factors[date][obj.factor_name]
            print(f'{obj.factor_name} calculated , valid_num is {factor_value.dropna().count()}')
            return obj.factor_name, factor_value

        if multi_thread:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_obj = {executor.submit(calculate_factor, obj, date): obj for obj in self.iter_instance(**kwargs)}
                for future in concurrent.futures.as_completed(future_to_obj):
                    obj = future_to_obj[future]
                    try:
                        factor_name, factor_value = future.result()
                        factor_values[factor_name] = factor_value
                    except Exception as e:
                        if ignore_error:
                            print(f'{obj.factor_name} generated an exception: {e}')
                        else:
                            raise e
        else:
            for obj in self.iter_instance(**kwargs):
                factor_name, factor_value = calculate_factor(obj , date)
                factor_values[factor_name] = factor_value

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

        if check_duplicates:
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
    def update_jobs(cls , start : int = -1 , end : int = 99991231 , overwrite = False , **kwargs):
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
        [obj.update_jobs(start , end , overwrite) for obj in self.iter_instance(**kwargs)]
        return self
    
    @staticmethod
    def Update(overwrite = False , show_progress = True , ignore_error = True):
        '''update factor data according to update jobs'''
        perform_update_jobs(overwrite , show_progress , ignore_error)

    @staticmethod
    def factor_filter(stock_factor_cls : Type['StockFactorCalculator'] , **kwargs):
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
