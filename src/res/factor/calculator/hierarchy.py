import pandas as pd

from itertools import combinations
from typing import Any , Literal , Type

from .factor_calc import StockFactorCalculator

from src.proj import PATH , MACHINE
from src.func.dynamic_import import dynamic_members
from src.func.parallel import parallel

class StockFactorHierarchy:
    '''hierarchy of factor classes'''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        assert PATH.fac_def.exists() , f'{PATH.fac_def} does not exist'
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
    
    @classmethod
    def export_factor_list(cls):
        if MACHINE.server:
            df = cls().factor_df()
            df.to_csv(PATH.rslt_factor.joinpath('factor_list.csv'))
    
    @classmethod
    def validate_category(cls , category0 : str , category1 : str):
        StockFactorCalculator.validate_category(category0 , category1)

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
        for level_path in PATH.fac_def.iterdir():
            if not level_path.is_dir(): continue
            level_name = level_path.stem
            if not level_name.startswith('level'): continue

            for name , obj in dynamic_members(level_path , subclass_of = StockFactorCalculator):
                assert name not in self.pool , f'{name} in module {obj.__module__} is duplicated'                        
                self.pool[name] = obj
                if level_name not in self.hier: self.hier[level_name] = []
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
        attr_list = ['level' , 'file_name' , 'factor_name' , 'init_date' , 'category0' , 'category1' , 'description' , 'min_date' , 'max_date']
        iterance = self.iter_instance(**kwargs)
        df_datas = []
        for cls in iterance: 
            attrs = [getattr(cls , a) for a in attr_list]
            df_datas.append(attrs)
        df = pd.DataFrame(df_datas, columns = attr_list)
        return df

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
                              multi_thread = True , ignore_error = True , verbose = True , **kwargs):
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
            parallel(calculate_factor , self.iter_instance(**kwargs) , factor_names , multi_thread , ignore_error = ignore_error)
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

    
