import pandas as pd

from itertools import combinations
from typing import Generator , Iterator , Type , Literal

from .factor_calc import FactorCalculator

from src.proj import PATH , MACHINE
from src.func.parallel import parallel

class StockFactorHierarchy:
    '''hierarchy of factor classes'''
    assert PATH.fac_def.exists() , f'{PATH.fac_def} does not exist'
    _instance = None
    initialized = False
    pool = FactorCalculator.registry

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.initialize()
    
    def __repr__(self):
        str_level_factors = [','.join(f'{level}({len(factors)})' for level , factors in self.hier.items())]
        return f'StockFactorHierarchy({str_level_factors})'
    
    def __iter__(self):
        '''return a generator of factor classes'''
        return self.iter_factors()
    
    def __getitem__(self , key : str):
        '''return a list of factor classes in a given level / or a factor class by factor_name'''
        return self.pool[key] if key in self.pool else self.hier[key]

    @classmethod
    def initialize(cls) -> None:
        """initialize the hierarchy of factor classes"""
        if cls.initialized:
            return
        FactorCalculator.import_definitions()
        cls.hier : dict[str , list[Type[FactorCalculator]]] = {}
        for obj in cls.pool.values():
            if obj.level not in cls.hier: 
                cls.hier[obj.level] = []
            cls.hier[obj.level].append(obj)
        cls.initialized = True

    @classmethod
    def load_factor_table(cls) -> pd.DataFrame:
        '''export factor stats to csv'''
        if MACHINE.server:
            df = pd.read_csv(PATH.local_shared.joinpath('factor_list.csv'))
        elif path := PATH.get_share_folder_path():
            df = pd.read_csv(path.joinpath('factor_list.csv'))
        else:
            df = pd.DataFrame()
        return df
    
    @classmethod
    def export_factor_table(cls) -> None:
        '''export factor list to csv'''
        if MACHINE.server:
            df = cls.full_factor_table()
            df.to_csv(PATH.local_shared.joinpath('factor_list.csv') , index = False)
            if path := PATH.get_share_folder_path():
                df.to_csv(path.joinpath('factor_list.csv') , index = False)

    @classmethod
    def full_factor_table(cls) -> pd.DataFrame:
        '''export factor stats to csv'''
        df = cls.factor_df().merge(cls.factor_stats(), on = 'factor_name' , how = 'left')
        return df

    @classmethod
    def factor_df(cls , **kwargs) -> pd.DataFrame:
        '''
        return a DataFrame of all factors with given attributes
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        meta_type : Literal['market' , 'stock' , 'affiliate' , 'pooling'] | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        attr_list = ['meta_type' , 'level' , 'factor_name' , 'init_date' , 'final_date' , 
        'file_name' , 'category0' , 'category1' , 'description' , 'min_date' , 'max_date']
        df_datas = []
        for calc in FactorCalculator.iter(**kwargs): 
            attrs = [getattr(calc , a) for a in attr_list]
            df_datas.append(attrs)
        df = pd.DataFrame(df_datas, columns = pd.Index(attr_list))
        return df

    @classmethod
    def factor_stats(cls , **kwargs) -> pd.DataFrame:
        '''
        return a DataFrame of all factors with given attributes
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        meta_type : Literal['market' , 'stock' , 'affiliate' , 'pooling'] | None = None
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        dfs = []
        for calc in FactorCalculator.iter(**kwargs): 
            daily_stats = calc.daily_stats()
            weekly_stats = calc.weekly_stats()
            if daily_stats.empty and weekly_stats.empty:
                continue
            df = pd.DataFrame({
                'factor_name' : [calc.factor_name],
                **cls.factor_mean_stats(daily_stats , 'daily'),
                **cls.factor_mean_stats(weekly_stats , 'weekly'),
            })
            dfs.append(df)
        return pd.concat(dfs)

    @classmethod
    def factor_mean_stats(cls , stats_df : pd.DataFrame , stats_type : Literal['daily' , 'weekly']) -> dict[str , float]:
        """return a DataFrame of all factors with given attributes"""
        ic = stats_df['ic'].mean()
        rankic = stats_df['rankic'].mean()
        gp = stats_df.filter(like='group@').sort_index(axis = 1 , key = lambda x: x.str.extract(r'group@(\d+)').astype(int)[0])
        gptop = gp.ffill(axis = 1).iloc[:,-1].mean()
        gpmid = gp.mean(axis = 1).mean()
        gpbot = gp.bfill(axis = 1).iloc[:,0].mean()
        stats = {
            f'{stats_type}_ic' : [ic],
            f'{stats_type}_rankic' : rankic,
            f'{stats_type}_long' : gptop - gpmid,
            f'{stats_type}_short' : gpmid - gpbot,
        }
        if stats_type == 'daily':
            stats['coverage'] = stats_df['coverage'].mean()
        return stats

    @classmethod
    def factor_names(cls) -> list[str]:
        '''return a list of factor names'''
        return [obj.factor_name for obj in cls().pool.values()]

    @classmethod
    def iter_levels(cls) -> Iterator[str]:
        '''return a list of levels'''
        return iter(cls().hier)
    
    @classmethod
    def iter_level_factors(cls , level : str) -> Generator[Type[FactorCalculator] , None , None]: 
        '''return a list of factor classes in a given level'''
        return (factor for factor in cls().hier[level])

    @classmethod
    def iter_factors(cls , **kwargs) -> Generator[Type[FactorCalculator] , None , None]:
        '''return a list of factor classes with given attributes'''
        return (factor for level in cls().iter_levels() for factor in cls().iter_level_factors(level))
    
    @classmethod
    def get_factor(cls , factor_name : str) -> Type[FactorCalculator]:
        '''
        return a factor class by factor_name
        e.g.
        factor_name = 'turn_12m'
        factor_cls = StockFactorHierarchy()[factor_name]
        '''
        return cls().pool[factor_name]
    
    def test_calc_all_factors(self , date : int = 20241031 , check_variation = True , check_duplicates = True , 
                              multi_thread = True , ignore_error = True , verbose = True , **kwargs) -> dict[str , pd.Series]:
        '''
        test calculation of all factors , if check_duplicates is True , check factors diffs' standard deviation and correlation
        factor_name : str | None = None
        level : str | None = None 
        file_name : str | None = None
        meta_type : Literal['market' , 'stock' , 'affiliate' , 'pooling'] | None = 'stock'
        category0 : str | None = None 
        category1 : str | None = None 
        '''
        
        def calculate_factor(obj : FactorCalculator):
            factor_value = obj.calc_factor(date)
            valid_ratio = len(factor_value.dropna()) / len(factor_value)
            if verbose or valid_ratio < 0.3: 
                print(f'{obj.factor_name} calculated , valid_ratio is {valid_ratio :.2%}')
            return factor_value

        kwargs = kwargs
        func_calls = {obj:(calculate_factor , {'obj' : obj}) for obj in FactorCalculator.iter(**kwargs)}
        
        factor_values : dict[str , pd.Series] = \
            parallel(func_calls , method = multi_thread , ignore_error = ignore_error)
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

    
