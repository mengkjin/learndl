import numpy as np
import pandas as pd
import warnings

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any , Literal

from src.proj import Proj , DB , Logger
from src.math import transform as T
from src.data import DataBlock , DATAVENDOR

from .alpha_model import AlphaModel
from .risk_model import RISK_MODEL
from .benchmark import Benchmark
from .universe import Universe

__all__ = ['StockFactor']

def append_indus(df : pd.DataFrame) -> pd.DataFrame:
    """
    append industry to factor dataframe
    """
    if df.empty:
        return df.assign(industry = [])
    old_index = df.index.names
    df = df.reset_index()
    secid = df['secid'].unique()
    date = df['date'].unique()
    appending = DATAVENDOR.risk_industry_exp(secid , date).to_dataframe().fillna('unknown')
    appending = pd.DataFrame(appending.idxmax(axis=1).rename('industry') , index = appending.index)
    df = df.merge(appending , on = ['secid','date']).set_index(old_index)
    return df

def append_ffmv(df : pd.DataFrame) -> pd.DataFrame:
    """
    append fmv to factor dataframe
    """
    if df.empty:
        return df.assign(ffmv = [])
    secid = df.index.get_level_values('secid').unique().to_numpy()
    date = df.index.get_level_values('date').unique().to_numpy()
    old_index = df.index.names
    appending = DATAVENDOR.ffmv(secid , date).to_dataframe()
    df = df.reset_index().merge(appending , on = ['secid','date']).set_index(old_index)
    return df

def append_fut_ret(df : pd.DataFrame , nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
    """
    append future return to factor dataframe
    example:
        df = append_fut_ret(df , nday = 10 , lag = 2 , ret_type = 'close')
    """
    if df.empty:
        return df.assign(ret = [])
    secid = df.index.get_level_values('secid').unique().to_numpy()
    date = df.index.get_level_values('date').unique().to_numpy()
    old_index = df.index.names
    appending = DATAVENDOR.nday_fut_ret(secid , date , nday , lag , ret_type = ret_type).to_dataframe()
    df = df.reset_index().merge(appending , on = ['secid','date']).set_index(old_index)
    return df

def melt_frame(df : pd.DataFrame) -> pd.DataFrame:
    """
    melt the dataframe from wide to long
    """
    if 'factor_name' in df.index.names:
        return df
    else:
        return df.melt(var_name = 'factor_name' , ignore_index = False).set_index('factor_name' , append=True)

def pivot_frame(df : pd.DataFrame) -> pd.DataFrame:
    """
    pivot the dataframe from long to wide
    """
    if 'factor_name' in df.index.names:
        return df.pivot_table(index = ['date' , 'secid'] , columns = 'factor_name' , values = 'value')
    else:
        return df

def whiten(df : pd.DataFrame | Any, ffmv_weighted = False , pivot = True) -> pd.DataFrame:
    """
    whiten the factors by date / factor_name , weight can be ffmv or not
    """
    df = melt_frame(df)
    if ffmv_weighted:
        df = append_ffmv(df)
        # df = df.groupby(by=['date' , 'factor_name'] , group_keys=False).apply(lambda x:whiten(x['value'] , weight = x['weight']))
        df = df.groupby(by=['date' , 'factor_name']).transform(lambda x:T.whiten(x['value'] , weight = x['weight']))
    else:
        df = df.groupby(by=['date' , 'factor_name']).transform(T.whiten)
    # if isinstance(df , pd.Series): df = df.to_frame()
    if pivot: 
        df = pivot_frame(df)
    return df

def winsor(df : pd.DataFrame | Any , pivot = True , **kwargs) -> pd.DataFrame:
    """
    winsorize the factors by date / factor_name
    """
    df = melt_frame(df)
    df = df.groupby(by=['date' , 'factor_name']).transform(T.winsorize , **kwargs)
    if isinstance(df , pd.Series): 
        df = df.to_frame()
    if pivot: 
        df = pivot_frame(df)
    return df

def fillna(df : pd.DataFrame | Any , 
           fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'zero' ,
           pivot = True , **kwargs) -> pd.DataFrame:
    """
    fill NA values of factors by date / factor_name
    """
    df = melt_frame(df)
    if fill_method == 'drop':
        df = df.dropna()
    elif fill_method == 'zero':
        df = df.fillna(0)
    elif fill_method == 'ffill':
        df = df.groupby(by=['date' , 'factor_name']).ffill()
    elif fill_method == 'mean':
        df = df.groupby(by=['date' , 'factor_name']).transform(lambda x: x.fillna(x.mean()))
    elif fill_method == 'median':
        df = df.groupby(by=['date' , 'factor_name']).transform(lambda x: x.fillna(x.median()))
    else:
        df = append_indus(df)  
        if fill_method == 'indus_mean':
            df = df.groupby(by=['date' , 'factor_name' , 'industry']).transform(lambda x: x.fillna(x.mean()))
        elif fill_method == 'indus_median':
            df = df.groupby(by=['date' , 'factor_name' , 'industry']).transform(lambda x: x.fillna(x.median()))
        else:
            raise ValueError(f'fill_method {fill_method} not supported')
    df = df.fillna(0)
    if isinstance(df , pd.Series): 
        df = df.to_frame()
    if pivot: 
        df = pivot_frame(df)
    return df

def neutralize(df : pd.DataFrame | Any , pivot = True , **kwargs) -> pd.DataFrame:
    """
    neutralize the factors by date / factor_name
    !! unrealized feature !!
    """
    if pivot: 
        df = pivot_frame(df)
    return df

def eval_grp_avg(
    x : pd.DataFrame , x_cols : list[str], y_name : str = 'ret', 
    group_num : int = 10 , excess = False , direction : int = 1
) -> pd.DataFrame:
    """
    evaluate the group average return of the factors
    """
    y = pd.DataFrame(x[y_name], index=x.index, columns=pd.Index([y_name]))
    group_bins = np.linspace(0,1,group_num + 1).tolist()
    group_labels = [i for i in range(1, group_num + 1)] if direction > 0 else [i for i in range(group_num, 0, -1)]

    rtn = list()
    for col in x_cols:
        y['group'] = pd.cut(x[col].rank(pct = True), bins=group_bins, labels=group_labels)
        if excess: 
            y[y_name] -= y[y_name].mean()
        grp_avg_ret = y.groupby('group' , observed = True)[y_name].mean().rename(col)
        rtn.append(grp_avg_ret)
    rtn = pd.concat(rtn, axis=1, sort=True)
    return rtn

def pnl_weights(x : pd.DataFrame, weight_type : str, direction : Any = 1 , group_num : int = 10) -> pd.Series | Any:
    """
    calculate the weights of the factors in calculation of weighted pnl
    """
    assert weight_type in ['long_short', 'top100' , 'long', 'short'] , weight_type
    assert np.all(np.sign(direction) != 0) , direction
    x = (x / np.nanstd(x, axis=0)) * np.sign(direction)
    def norm_weight(xx : pd.Series | Any) -> pd.Series | Any:
        return (xx / np.sum(xx, axis=0))
    eq_wgt = 1 / x.count(numeric_only=True)
    if weight_type == 'top100':
        wgt = norm_weight(x > x.quantile(1 - 100 / x.shape[0])) - eq_wgt
    elif weight_type == 'long':
        wgt = norm_weight(x > x.quantile(1 - 1 / group_num)) - eq_wgt
    elif weight_type == 'short':
        wgt = -1 * (norm_weight(x < x.quantile(1 / group_num)) - eq_wgt)
    elif weight_type == 'long_short':
        wgt = norm_weight(x > x.quantile(1 - 1 / group_num)) - norm_weight(x < x.quantile(1 / group_num))
    return wgt

def eval_weighted_pnl(x : pd.DataFrame , weight_type : str , direction : Any , group_num = 10 , y_name = 'ret') -> float:
    """
    evaluate the weighted pnl (top - bottom) of the factors
    """
    vals = x.drop(columns=[y_name])
    rets = x[[y_name]].to_numpy()
    weights = pnl_weights(vals, weight_type, direction, group_num)
    rtn = (weights * rets).sum(axis = 0)
    return rtn

def param_match(args1 : dict[str,Any] , args2 : dict[str,Any]) -> bool:
    """
    check if the parameters match each other perfectly
    """
    if len(args1) != len(args2): 
        return False
    matches = [k in args2 and args2[k] == v for k , v in args1.items()]
    return all(matches) 

class FactorStats:
    """
    FactorStats class is used to store and manipulate factor statistics
    """
    available_stats : list[str] = ['ic' , 'ic_indus' , 'group_perf' , 'weighted_pnl' , 'coverage']
    unique_keys : dict[str,list[str]] = {
        'ic' : ['date' , 'factor_name'],
        'ic_indus' : ['date' , 'factor_name' , 'industry'],
        'group_perf' : ['date' , 'factor_name' , 'group'],
        'weighted_pnl' : ['weight_type' , 'date'],
        'coverage' : ['date']}
    def __init__(self , name : str):
        assert name in self.available_stats , f'name {name} is not in {self.available_stats}'
        self.name = name
        self.stats : dict[str,pd.DataFrame] = {}
    
    def __repr__(self):
        return f'FactorStats(name={self.name}, stats={list(self.stats.keys())})'
    
    def __str__(self):
        return self.__repr__()

    @staticmethod
    def param_to_str(params : dict[str,Any]) -> str:
        return '.'.join([f'{k}={v}' for k , v in params.items()])

    @classmethod
    def match_params(cls , params1 : dict[str,Any] , params2 : dict[str,Any]) -> bool:
        return cls.param_to_str(params1) == cls.param_to_str(params2)

    def update_stat(self , params : dict[str,Any] , stat : pd.DataFrame):
        self.stats[self.param_to_str(params)] = stat

    def append_stat(self , params : dict[str,Any] , stat : pd.DataFrame , keys : list[str] | None = None):
        stat = pd.concat([self.get_stat(params) , stat])
        index_names = [key for key in stat.index.names if key is not None]
        if index_names:
            stat = stat.reset_index(drop=False)
        if keys is None:
            keys = self.unique_keys[self.name]
        if keys:
            stat = stat.drop_duplicates(subset=keys , keep='last').sort_values(by=keys)
        if index_names:
            stat = stat.set_index(index_names)
        self.update_stat(params , stat)

    def get_stat(self , params : dict[str,Any]) -> pd.DataFrame:
        return self.stats.get(self.param_to_str(params) , pd.DataFrame())

    def dates_not_in_stat(self , params : dict[str,Any] , dates : np.ndarray) -> np.ndarray:
        stat = self.get_stat(params)
        if stat.empty:
            stat_dates = np.array([])
        else:
            stat_dates = stat.index.get_level_values('date').unique() if 'date' in stat.index.names else stat['date'].unique()
        return np.setdiff1d(dates, stat_dates)

    def load(self , path : Path) -> int:
        if not path.exists():
            return 0
        assert path.is_dir() , f'path {path} is not a directory'
        files = list(path.glob('*.feather'))
        for file in files:
            self.stats[file.stem] = DB.load_df(file)
        return len(files)

    def save(self , path : Path) -> int:
        path.mkdir(parents=True , exist_ok=True)
        for name , stat in self.stats.items():
            DB.save_df(stat , path.joinpath(f'{name}.feather') , vb_level = 99)
        return len(self.stats)

    @classmethod
    def create_cache_factor_stats(cls) -> dict[str,'FactorStats']:
        return {name:FactorStats(name) for name in cls.available_stats}

class CacheFactorStats:
    """
    CacheFactorStats class is used to store and manipulate multiple factor statistics
    """
    def __init__(self):
        self.factor_stats = FactorStats.create_cache_factor_stats()

    def __getitem__(self , name : str) -> FactorStats:
        return self.factor_stats[name]

    def load(self , path : Path | None):
        if path is None or not path.exists():
            return
        stats_num = sum([factor_stats.load(path.joinpath(factor_stats.name)) for factor_stats in self.factor_stats.values()])
        Logger.success(f'Load {stats_num} Factor Stats from {path}' , indent = 0 , vb_level = Proj.vb.max)
        return self

    def save(self , path : Path | None):
        if path is None:
            return
        stats_num = sum([factor_stats.save(path.joinpath(factor_stats.name)) for factor_stats in self.factor_stats.values()])
        Logger.success(f'Save {stats_num} Factor Stats to {path}' , indent = 0 , vb_level = Proj.vb.max)

    @property
    def ic(self) -> FactorStats:
        return self.factor_stats['ic']
    
    @property
    def ic_indus(self) -> FactorStats:
        return self.factor_stats['ic_indus']
    
    @property
    def group_perf(self) -> FactorStats:
        return self.factor_stats['group_perf']
    
    @property
    def weighted_pnl(self) -> FactorStats:
        return self.factor_stats['weighted_pnl']
    
    @property
    def coverage(self) -> FactorStats:
        return self.factor_stats['coverage']


class StockFactor:
    """
    StockFactor class is used to store and manipulate factor data
    factor data can be a pandas DataFrame , a pandas Series , a DataBlock , a StockFactor , or a dictionary of pd.Series
    can include multiple factors

    example:
        factor = StockFactor(factor_data)
        factor.normalize()
        factor.analyze()
        factor.eval_ic()
        factor.eval_group_perf()
    """
    def __new__(cls , factor = None , *args , **kwargs):
        if not isinstance(factor , StockFactor):
            factor = super().__new__(cls)
        Proj.States.factor = factor
        return factor

    def __init__(self , factor : 'None|pd.DataFrame|pd.Series|DataBlock|StockFactor|dict[int,pd.Series]' = None , * ,
                 normalized : bool | None = None , 
                 benchmark : Benchmark | str | None = None , 
                 cache_factor_stats : CacheFactorStats | None = None ,
                 factor_names : list[str] | None = None):
        if factor is None:
            assert factor_names , 'factor_names must be provided if factor input is None'
            factor = pd.DataFrame(columns=['date' , 'secid' , *factor_names])
        self.normalized = False
        self.benchmark = None
        
        self.cache_alpha_models : dict[str,AlphaModel] = {}
        self.cache_benchmark_subsets : dict[str,StockFactor] = {}
        self.cache_factor_stats : CacheFactorStats = cache_factor_stats if cache_factor_stats else CacheFactorStats()
        self.update(factor , normalized , benchmark)

    def __repr__(self):
        format_str = f'{self.__class__.__name__}(normalized={self.normalized},names={self.factor_names},dates={{}})'
        if self.date.size == 0:
            return format_str.format('None')
        else:
            return format_str.format(f'{self.date.size}({self.date.min()}-{self.date.max()})')
    
    def __call__(self , benchmark):
        return self.within(benchmark)

    @property
    def empty(self) -> bool:
        """
        return True if the factor is empty
        """
        return self.prior_input.empty

    def update(self , factor : 'pd.DataFrame|pd.Series|DataBlock|StockFactor|dict[int,pd.Series]' , 
               normalized : bool | None = None , benchmark : Benchmark | str | None = None):
        """
        update the factor data
        """
        if isinstance(factor , StockFactor):
            if normalized is not None:
                assert factor.normalized == normalized , f'normalized must be the same as the original factor : {factor.normalized} != {normalized}'
            if benchmark is not None:
                assert factor.benchmark == benchmark.name if isinstance(benchmark , Benchmark) else benchmark , f'benchmark must be the same as the original factor : {factor.benchmark} != {benchmark}'
            return factor

        if isinstance(factor , dict):
            factor = pd.concat([(f.to_frame() if isinstance(f , pd.Series) else f).assign(date = d) 
                                for d , f in factor.items() if not f.empty])
        elif isinstance(factor , pd.Series):
            factor = factor.to_frame()
            
        assert isinstance(factor , (pd.DataFrame , DataBlock)) , f'factor must be a pandas DataFrame or DataBlock , but got {type(factor)} : {factor}'

        if isinstance(factor , pd.DataFrame):
            factor = factor.reset_index().drop(columns=['index'] , errors='ignore')
            if 'date' in factor.columns: 
                factor = factor.set_index('date' , append=True)
            if 'secid' in factor.columns: 
                factor = factor.set_index('secid' , append=True)
            if None in factor.index.names:
                factor = factor.reset_index([None] , drop=True)
            self.input_df = factor
        elif isinstance(factor , DataBlock):
            self.input_blk = factor
        else:
            raise TypeError(f'Unknown factor type: {type(factor)}')

        if normalized is not None:
            self.normalized = normalized
        if benchmark is not None:
            self.benchmark = benchmark.name if isinstance(benchmark , Benchmark) else benchmark

        return self

    @property
    def input_df(self) -> pd.DataFrame | None:
        """
        return the input DataFrame of the factor
        """
        if not hasattr(self , '_df'):
            self._df = None
        return self._df

    @input_df.setter
    def input_df(self , df : pd.DataFrame | None):
        """set the input DataFrame of the factor"""
        self._df = df
    
    @property
    def input_blk(self) -> DataBlock | None:
        """
        return the input DataBlock of the factor
        """
        if not hasattr(self , '_blk'):
            self._blk = None
        return self._blk
    
    @input_blk.setter
    def input_blk(self , blk : DataBlock | None):
        """set the input DataBlock of the factor"""
        self._blk = blk

    def copy(self): 
        """
        return a copy of the factor
        """
        return deepcopy(self)

    def filter_dates(self , dates : np.ndarray | Any | None = None , exclude = False , inplace = False):
        """
        filter the factor data by dates or other index
        """
        if self.empty or dates is None:
            return self
        df = self.frame().query('date not in @dates' if exclude else 'date in @dates')
        return self.update(df) if inplace else StockFactor(df , normalized = self.normalized , benchmark = self.benchmark)

    def filter_dates_between(self , start_dt : int , end_dt : int , inplace = False):
        """
        filter the factor data by dates between start_dt and end_dt
        """
        dates = self.date
        dates = dates[(dates >= start_dt) & (dates <= end_dt)]
        return self.filter_dates(dates , inplace = inplace)

    def filter_secid(self , secid : np.ndarray | Any | None = None , exclude = False , inplace = False):
        if self.empty or secid is None:
            return self
        df = self.frame().query('secid not in @secid' if exclude else 'secid in @secid')
        return self.update(df) if inplace else StockFactor(df , normalized = self.normalized , benchmark = self.benchmark)

    def filter_by(self , step : int = 1):
        """
        filter the factor data by step
        """
        return self.filter_dates(self.date[::step])

    def rename(self , new_name : str | list[str] | dict[str,str] , inplace = True):
        """
        rename the factor data
        """
        if not inplace: 
            self = self.copy()
        if isinstance(new_name , str):
            assert self.factor_num == 1 , f'only one factor is supported for using str for renaming : {self.factor_names}'
            mapping = {self.factor_names[0]:new_name}
        elif isinstance(new_name , list):
            assert len(new_name) == self.factor_num , f'the length of new_name must be equal to the number of factors : {self.factor_names}'
            mapping = {self.factor_names[i]:new_name[i] for i in range(len(new_name))}
        elif isinstance(new_name , dict):
            mapping = new_name
        if self.input_df is not None: 
            self.input_df.rename(columns=mapping , inplace=True)
        if self.input_blk is not None: 
            self.input_blk.rename_feature(mapping)
        return self

    def join(self , *others : 'StockFactor'):
        """
        join the factor with other factors
        """
        df = self.frame()
        for other in others: 
            df = df.join(other.frame() , how = 'outer')
        return StockFactor(df)
    
    def ew(self):
        """
        calculate the equal-weighted factor from multiple factors
        """
        if self.factor_num == 1:
            return self
        else:
            df : pd.Series | Any = self.frame().mean(axis = 1)
            return StockFactor(df.rename('multifactor_ew'))

    def frame(self) -> pd.DataFrame:
        """
        return the factor data as a pandas DataFrame
        """
        prior_input = self.prior_input
        if isinstance(prior_input , DataBlock):
            self.input_df = prior_input.to_dataframe()
            return self.input_df
        else:
            assert prior_input is not None , 'prior_input cannot be None'
            return prior_input
    
    def block(self) -> DataBlock:
        """
        return the factor data as a DataBlock
        """
        prior_input = self.prior_input
        if isinstance(prior_input , DataBlock):
            return prior_input
        else:
            assert prior_input is not None , 'prior_input cannot be None'
            self.input_blk = DataBlock.from_dataframe(prior_input)
            return self.input_blk

    @property
    def secid(self) -> np.ndarray: 
        """
        return the unique secid of the factor
        """
        prior_input = self.prior_input
        if isinstance(prior_input , DataBlock):
            return prior_input.secid
        elif prior_input.empty:
            return np.array([])
        else:
            return prior_input.index.get_level_values('secid').unique().to_numpy()

    @property
    def date(self) -> np.ndarray:
        """
        return the unique date of the factor
        """
        if self.pseudo_date is not None:
            return self.pseudo_date
        else:
            return self.data_date

    @property
    def data_date(self) -> np.ndarray:
        prior_input = self.prior_input
        if isinstance(prior_input , DataBlock):
            return prior_input.date
        elif prior_input.empty:
            return np.array([])
        else:
            return prior_input.index.get_level_values('date').unique().to_numpy()

    @property
    def pseudo_date(self) -> np.ndarray | None:
        """
        return the pseudo date of the factor
        """
        if not hasattr(self , '_pseudo_date'):
            return None
        return self._pseudo_date

    @pseudo_date.setter
    def pseudo_date(self , date : np.ndarray | None):
        if date is not None:
            Logger.alert1(f'Setting {self} pseudo date to {date}' , indent = 1 , vb_level = Proj.vb.max)
            Logger.alert1(f'Original date : {self.data_date}' , indent = 1 , vb_level = Proj.vb.max)
        self._pseudo_date = date

    def set_pseudo_date(self , date : np.ndarray | None = None):
        self.pseudo_date = date
        
    @property
    def factor_names(self) -> np.ndarray:
        """
        return the factor names of the factor
        """
        prior_input = self.prior_input
        if isinstance(prior_input , DataBlock):
            return prior_input.feature
        else:
            return prior_input.columns.to_numpy()
        
    @property
    def factor_num(self) -> int:
        """
        return the number of factors
        """
        return self.prior_input.shape[-1]

    @property
    def prior_input(self) -> pd.DataFrame | DataBlock:
        """
        return the prior input of the factor (DataFrame or DataBlock that is initialized with)
        """
        if self.input_df is not None:
            return self.input_df
        else:
            assert self.input_blk is not None , 'input_blk and input_df cannot be both None'
            return self.input_blk
        
    @classmethod
    def Load(cls , factor_name : str , date : int):
        """
        load the factor data from the database by factor name and date
        """
        df = DB.load('factor' , factor_name , date).assign(date = date)
        return cls(df)

    @classmethod
    def Loads(cls , factor_name : str , start : int | None = None , end : int | None = None , dates : np.ndarray | None = None):
        """
        load the factor data from the database by factor name and date range
        """
        df = DB.loads('factor' , factor_name , dates = dates , start_dt=start , end_dt=end)
        return cls(df)
        
    def select(self , secid = None , date = None , factor_name = None):
        """
        select the factor data by secid , date , factor name
        """
        prior_input = self.prior_input
        if isinstance(prior_input , pd.DataFrame):
            df = prior_input.reset_index()
            if date is not None:  
                date = date if isinstance(date , Iterable) else [date]
                df = df.query('date in @date')
            if secid is not None: 
                secid = secid if isinstance(secid , Iterable) else [secid]
                df = df.query('secid in @secid')
            df = df.set_index(['date' , 'secid'])
            if factor_name is not None: 
                df = df[factor_name]
            return StockFactor(df)
        else:
            return StockFactor(prior_input.align(secid , date , factor_name))

    def within(self , benchmark : Benchmark | str | None , use_cache = True , share_cache_factor_stats = True) -> 'StockFactor':
        """
        use benchmark to mask factor , only keep the factors that are in the benchmark
        """
        if isinstance(benchmark , str): 
            benchmark = Benchmark(benchmark)
        if not benchmark: 
            return self
        if benchmark.name not in self.cache_benchmark_subsets or not use_cache:
            factor_input = benchmark(self.prior_input) if benchmark else self.prior_input
            cache_factor_stats = self.cache_factor_stats if share_cache_factor_stats else None
            subset = StockFactor(factor_input , normalized = self.normalized , benchmark = benchmark.name , cache_factor_stats = cache_factor_stats)
            self.cache_benchmark_subsets[benchmark.name] = subset
        return self.cache_benchmark_subsets[benchmark.name]
    
    def alpha_model(self , use_cache = True) -> AlphaModel:
        """
        transform the factor to alpha model , only one factor is supported
        """
        assert len(self.factor_names) == 1 , f'only one factor is supported for alpha model , but got {len(self.factor_names)}'
        name = self.factor_names[0]
        if name not in self.cache_alpha_models or not use_cache:
            self.cache_alpha_models[name] = self._get_alpha_model(name)
        return self.cache_alpha_models[name]

    def alpha_models(self , use_cache = True) -> list[AlphaModel]:
        """
        transform the factor to alpha models , multiple factors are supported
        """
        models = []
        for name in self.factor_names:
            if name not in self.cache_alpha_models or not use_cache:
                self.cache_alpha_models[name] = self._get_alpha_model(name)
            models.append(self.cache_alpha_models[name])
        return models

    def risk_model(self , load = True):
        """
        load the risk model for the factor
        """
        return RISK_MODEL.load_models(self.date if load else None)

    def universe(self , name : str = 'top-1000' , load = True):
        """
        get the universe for the factor
        """
        univ = Universe(name)
        univ.to_portfolio(self.date if load else None)
        return univ

    def within_benchmarks(self , load = True):
        """
        pre-load benchmarked factors
        """
        return [self.within(bm) for bm in Benchmark.TESTS]

    def day_returns(self , load = True):
        """
        get the daily quotes for the factor
        """
        if len(self.date) == 0:
            return DataBlock()
        DATAVENDOR.TRADE.load_trd_within(self.date.min() , self.date.max() , overwrite = False)
        return DATAVENDOR.get_quotes_block(self.date if load else None , extend = 60)

    def day_quotes(self , load = True):
        """
        get the daily quotes for the factor
        """
        return DATAVENDOR.TRADE.loads(self.date if load else None , 'trd')

    def _get_alpha_model(self , name : str):
        """
        update the alpha models
        """
        assert name in self.factor_names , f'{name} is not in the factor names : {self.factor_names}'
        return AlphaModel.from_dataframe(self.frame()[name] , name)

    def frame_with_cols(self , indus = False , fut_ret = False , ffmv = False ,
                        nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close' ,
                        dates : np.ndarray | None = None) -> pd.DataFrame:
        """
        return the factor data with additional columns
        """
        df = self.frame()
        if dates is not None:
            df = df.query('date in @dates')
        if indus:   
            df = append_indus(df)
        if fut_ret: 
            df = append_fut_ret(df , nday , lag , ret_type)
        if ffmv:    
            df = append_ffmv(df)
        return df   

    def eval_ic(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                ret_type : Literal['close' , 'vwap'] = 'close' , use_cache = True , all_dates = False) -> pd.DataFrame:
        """
        evaluate the IC of the factor
        """
        params = {'bm' : str(self.benchmark) , 'n' : nday , 'lag' : lag , 'ic' : ic_type , 'ret' : ret_type}
        calc_dates = self.cache_factor_stats.ic.dates_not_in_stat(params , self.date) if use_cache else self.date
        if len(calc_dates) > 0:
            df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type , dates = calc_dates)
            grouped = df.groupby(by=['date'], as_index=True)
            def df_ic(subdf : pd.DataFrame , **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='An input array is constant; the correlation coefficient is not defined' , category=RuntimeWarning)
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide' , category=RuntimeWarning)
                    return subdf[self.factor_names].corrwith(subdf['ret'], method=ic_type)
            ic = grouped.apply(df_ic , include_groups = False).rename_axis('factor_name',axis='columns')
            self.cache_factor_stats.ic.append_stat(params , ic , keys = ['date'])
        stat = self.cache_factor_stats.ic.get_stat(params)
        return stat if all_dates else stat.query('date in @self.date')
    
    def eval_ic_indus(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                      ret_type : Literal['close' , 'vwap'] = 'close' , use_cache = True , all_dates = False) -> pd.DataFrame:
        """
        evaluate the IC of the factor by industry
        """
        params = {'bm' : str(self.benchmark) , 'n' : nday , 'lag' : lag , 'ic' : ic_type , 'ret' : ret_type}
        calc_dates = self.cache_factor_stats.ic_indus.dates_not_in_stat(params , self.date) if use_cache else self.date
        if len(calc_dates) > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*correlation coefficient is not defined.*')
                df = self.frame_with_cols(indus = True , fut_ret = True , nday = nday , lag = lag , ret_type = ret_type , dates = calc_dates)
                def df_ic(subdf : pd.DataFrame , **kwargs):
                    return subdf[self.factor_names].corrwith(subdf['ret'], method=ic_type)
                ic = df.groupby(['date' , 'industry']).apply(df_ic , include_groups = False).\
                    rename_axis('factor_name',axis='columns').reset_index(drop=False).\
                    melt(id_vars = ['date' , 'industry'] , var_name = 'factor_name' , value_name = 'ic_indus')
            self.cache_factor_stats.ic_indus.append_stat(params , ic , keys = ['date' , 'industry'])
        stat = self.cache_factor_stats.ic_indus.get_stat(params)
        return stat if all_dates else stat.query('date in @self.date')
    
    @staticmethod
    def _eval_group_perf(df : pd.DataFrame , factors , group_num : int = 10 , excess = False , direction : int = 1) -> pd.DataFrame:
        """
        evaluate the group performance df , columns must have : ['ret' , *factors]
        """
        kwargs = {'x_cols' : factors , 'y_name' : 'ret' , 'group_num' : group_num , 'excess' : excess , 'include_groups' : False , 'direction' : direction}
        df = df.groupby('date').apply(eval_grp_avg , **kwargs) 
        df = df.melt(var_name='factor_name' , value_name='group_ret' , ignore_index=False).sort_values(['date','factor_name']).reset_index()
        return df

    def eval_group_perf(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 , excess = False , 
                        ret_type : Literal['close' , 'vwap'] = 'close' , use_cache = True , all_dates = False) -> pd.DataFrame:
        """
        evaluate the group performance of the factor
        """
        params = {'bm' : str(self.benchmark) , 'n' : nday , 'lag' : lag , 'gp' : group_num , 'exc' : excess , 'ret' : ret_type}
        calc_dates = self.cache_factor_stats.group_perf.dates_not_in_stat(params , self.date) if use_cache else self.date
        if len(calc_dates) > 0:
            df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type , dates = calc_dates)
            # assert not np.any(np.isinf(df['ret'])), f'inf values in factor data ret : {df[np.isinf(df).any(axis=1)]}'
            df = self._eval_group_perf(df , self.factor_names , group_num , excess)
            df['start'] = DATAVENDOR.td_array(df['date'] , lag)
            df['end']   = DATAVENDOR.td_array(df['date'] , lag + nday - 1)
            self.cache_factor_stats.group_perf.append_stat(params , df , keys = ['date' , 'factor_name' , 'group'])
        stat = self.cache_factor_stats.group_perf.get_stat(params)
        return stat if all_dates else stat.query('date in @self.date')
    
    def eval_weighted_pnl(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                          ret_type : Literal['close' , 'vwap'] = 'close' , direction : Literal[1,0,-1] = 0 ,
                          use_cache = True , all_dates = False) -> pd.DataFrame:
        """
        evaluate the weighted pnl of the factor
        """
        params = {'bm' : str(self.benchmark) , 'n' : nday , 'lag' : lag , 'gp' : group_num , 'ret' : ret_type , 'sign' : direction}
        calc_dates = self.cache_factor_stats.weighted_pnl.dates_not_in_stat(params , self.date) if use_cache else self.date
        if len(calc_dates) > 0:
            df = self.frame_with_cols(fut_ret=True , nday=nday , lag=lag , ret_type=ret_type , dates=calc_dates)
            dr = np.sign(df.corr().loc['ret'].drop('ret')) if direction == 0 else direction

            dfs : list[pd.DataFrame] = []
            for wt in ['long' , 'long_short' , 'short']:
                kwargs = {'weight_type' : wt , 'direction' : dr , 'group_num' : group_num , 'include_groups' : False}
                df_wt = df.groupby('date').apply(eval_weighted_pnl, **kwargs).\
                    reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ret').assign(weight_type = wt)
                dfs.append(df_wt)

            pnl = pd.concat(dfs).set_index(['weight_type' , 'date'])
            self.cache_factor_stats.weighted_pnl.append_stat(params , pnl , keys = ['weight_type' , 'date'])
        stat = self.cache_factor_stats.weighted_pnl.get_stat(params)
        return stat if all_dates else stat.query('date in @self.date')

    def coverage(self , benchmark : Benchmark | str | None = None , use_cache = True , all_dates = False) -> pd.DataFrame:
        """
        evaluate the coverage of the factor by benchmark
        """
        params = {'bm' : str(benchmark) if isinstance(benchmark,Benchmark) else str(benchmark)}
        calc_dates = self.cache_factor_stats.coverage.dates_not_in_stat(params , self.date) if use_cache else self.date
        if len(calc_dates) > 0:
            if isinstance(benchmark , str) or benchmark is None: 
                benchmark = Benchmark(benchmark)
            factor = self.within(benchmark)
            benchmark_size = pd.Series(benchmark.sec_num(calc_dates) , index = calc_dates)
            coverage = factor.frame().groupby('date').apply(lambda x:x.dropna().count(numeric_only=True))
            for factor_name in coverage:
                coverage[factor_name] = (coverage[factor_name] / benchmark_size).clip(lower=0 , upper=1)
            self.cache_factor_stats.coverage.append_stat(params , coverage , keys = ['date'])
        stat = self.cache_factor_stats.coverage.get_stat(params)
        return stat if all_dates else stat.query('date in @self.date')

    def time_series_stats(self , nday : int = 1 , lag : int = 1) -> pd.DataFrame:
        """
        evaluate the period time series stats of the factor
        """
        ic = self.eval_ic(nday, lag , ic_type = 'pearson').rename(columns = lambda x:f'ic')
        rankic = self.eval_ic(nday, lag , ic_type = 'spearman').rename(columns = lambda x:f'rankic')
        
        gp = self.eval_group_perf(nday, lag).\
            pivot_table(index = ['date'] , values = 'group_ret' , columns = 'group' , observed=True).\
            rename(columns = lambda x:f'group@{x}')
        ii = self.eval_ic_indus(nday, lag).\
            pivot_table(index = ['date'] , values = 'ic_indus' , columns = 'industry' , observed=True).\
            rename(columns = lambda x:f'ic_indus@{x}')
        cv = self.coverage().rename(columns = lambda x:f'coverage')
        return ic.join(rankic).join(gp).join(ii).join(cv).reset_index(drop=False)

    def daily_stats(self) -> pd.DataFrame:
        """
        evaluate the daily stats of the factor
        """
        return self.time_series_stats(1, 1)

    def weekly_stats(self) -> pd.DataFrame:
        """
        evaluate the weekly stats of the factor
        """
        return self.time_series_stats(5, 1)
    
    def normalize(self , fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
                  weighted_whiten = False , order = ['fillna' , 'winsor' , 'whiten'] , inplace = False):
        """
        normalize the factor data by fill method , weighted whiten , and winsorize
        can specify the order of the steps
        """
        if not order:
            return self
        df = self.normalize_df(self.frame() , fill_method = fill_method , weighted_whiten = weighted_whiten , order = order)
        if inplace: 
            return self.update(df , normalized = True)
        else:
            return StockFactor(df , normalized = True , benchmark = self.benchmark)
    
    def select_analytic(self , task_name : str , **kwargs):
        """
        select the analytic task by task name
        """
        from src.res.factor.analytic import FactorPerfTest

        match_task = [task for task in FactorPerfTest.TASK_LIST if task.match_name(task_name)]
        assert match_task and len(match_task) <= 1 , f'no match or duplicate match tasks : {task_name}'
        task , task_name = match_task[0] , match_task[0].__name__
        if not hasattr(self , 'analytic_tasks'): 
            self.analytic_tasks = {}
        if task_name not in self.analytic_tasks: 
            self.analytic_tasks[task_name] = task(**kwargs)
        return self.analytic_tasks[task_name]

    def analyze(self , 
                task_name : Literal['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Decay', 'IC_Indus',
                                    'IC_Year','IC_Benchmark','IC_Monotony','PnL_Curve',
                                    'Style_Corr','Group_Curve','Group_Decay','Group_IR_Decay',
                                    'Group_Year','Distrib_Curve'] | str , 
                nday : int = 5 ,
                plot = True , display = True , **kwargs):
        """
        analyze the factor by task name (only one task is supported)
        access by self.analytic_tasks[task_name]
        """
        task = self.select_analytic(task_name ,  nday = nday , **kwargs)
        task.calc(self)
        if plot: 
            task.plot(show = display)
        return self

    def fast_analyze(self , task_list = ['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Benchmark','IC_Monotony','Style_Corr'] , nday : int = 5 , **kwargs):
        """
        fast analyze the factor
        default task list is ['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Benchmark','IC_Monotony','Style_Corr']
        access by self.analytic_tasks[task_name]
        """
        for task_name in task_list: 
            self.analyze(task_name , nday = nday , **kwargs)
        return self

    def full_analyze(self , nday : int = 5 , **kwargs):
        """
        full analyze the factor , 
        default task list is ['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Decay', 'IC_Indus',
                     'IC_Year','IC_Benchmark','IC_Monotony','PnL_Curve',
                     'Style_Corr','Group_Curve','Group_Decay','Group_IR_Decay',
                     'Group_Year','Distrib_Curve']
        access by self.analytic_tasks[task_name]
        """
        task_list = ['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Decay', 'IC_Indus',
                     'IC_Year','IC_Benchmark','IC_Monotony','PnL_Curve',
                     'Style_Corr','Group_Curve','Group_Decay','Group_IR_Decay',
                     'Group_Year','Distrib_Curve']
        for task_name in task_list: 
            self.analyze(task_name , nday = nday , **kwargs)
        return self

    @staticmethod
    def fillna(df : pd.DataFrame , fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop') -> pd.DataFrame:
        """
        fill na values of the factor data
        """
        return fillna(df , fill_method = fill_method)

    @staticmethod
    def normalize_df(df : pd.DataFrame , fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
                  weighted_whiten = False , order = ['fillna' , 'winsor' , 'whiten'] , inplace = False):
        """
        normalize the dataframe factor data by fill method , weighted whiten , and winsorize
        can specify the order of the steps
        """
        if df.empty:
            return df
        if 'date' not in df.index.names:
            df = df.set_index('date' , append=True)
        if 'secid' not in df.index.names:
            df = df.set_index('secid' , append=True)
        if None in df.index.names:
            df = df.reset_index([None] , drop=True)
        assert 'date' in df.index.names and 'secid' in df.index.names , f'df must have date and secid as index : {df}'
        for step in order:
            if step == 'fillna':   
                df = fillna(df , fill_method = fill_method)
            elif step == 'winsor': 
                df = winsor(df)
            elif step == 'whiten': 
                df = whiten(df , ffmv_weighted = weighted_whiten)
            else:
                raise ValueError(f'step {step} not supported')
        df = pivot_frame(df).reset_index(['date' , 'secid']).reset_index(drop=True).rename_axis(None , axis = 1)
        return df