import numpy as np
import pandas as pd
import warnings

from collections.abc import Iterable
from copy import deepcopy
from typing import Any , Literal

from src.proj import Proj , DB
from src.math import transform as T
from src.data import DataBlock , DATAVENDOR

from .alpha_model import AlphaModel
from .risk_model import RISK_MODEL
from .benchmark import Benchmark
from .universe import Universe

__all__ = ['StockFactor']

def append_indus(df : pd.DataFrame):
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

def append_ffmv(df : pd.DataFrame):
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

def append_fut_ret(df : pd.DataFrame , nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close'):
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

def melt_frame(df : pd.DataFrame):
    """
    melt the dataframe from wide to long
    """
    if 'factor_name' in df.index.names:
        return df
    else:
        return df.melt(var_name = 'factor_name' , ignore_index = False).set_index('factor_name' , append=True)

def pivot_frame(df : pd.DataFrame):
    """
    pivot the dataframe from long to wide
    """
    if 'factor_name' in df.index.names:
        return df.pivot_table(index = ['date' , 'secid'] , columns = 'factor_name' , values = 'value')
    else:
        return df

def whiten(df : pd.DataFrame | Any, ffmv_weighted = False , pivot = True):
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

def winsor(df : pd.DataFrame | Any , pivot = True , **kwargs):
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
           pivot = True , **kwargs):
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

def neutralize(df : pd.DataFrame | Any , pivot = True , **kwargs):
    """
    neutralize the factors by date / factor_name
    !! unrealized feature !!
    """
    if pivot: 
        df = pivot_frame(df)
    return df

def eval_grp_avg(x : pd.DataFrame , x_cols : list[str], y_name : str = 'ret', group_num : int = 10 , excess = False , direction : int = 1) -> pd.DataFrame:
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

def pnl_weights(x : pd.DataFrame, weight_type : str, direction : Any = 1 , group_num : int = 10):
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

def eval_weighted_pnl(x : pd.DataFrame , weight_type : str , direction : Any , group_num = 10 , y_name = 'ret'):
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
        if isinstance(factor , StockFactor):
            return factor
        else:
            return super().__new__(cls)

    def __init__(self , factor : 'None|pd.DataFrame|pd.Series|DataBlock|StockFactor|dict[int,pd.Series]' = None , normalized = False , factor_names : list[str] | None = None):
        if factor is None:
            factor = pd.DataFrame()
        self.update(factor , normalized , factor_names)
        Proj.States.factor = self

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

    def update(self , factor : 'pd.DataFrame|pd.Series|DataBlock|StockFactor|dict[int,pd.Series]' , normalized = False , factor_names : list[str] | None = None):
        """
        update the factor data
        """
        factor_names = factor_names or []
        if isinstance(factor , StockFactor):
            assert factor.normalized == normalized , f'normalized must be the same as the original factor : {factor.normalized} != {normalized}'
            return factor

        if isinstance(factor , dict):
            if not factor:
                factor = pd.DataFrame(columns=['date' , 'secid' , *factor_names])
            else:
                factor = pd.concat([(f.to_frame() if isinstance(f , pd.Series) else f).assign(date = d) 
                                    for d , f in factor.items() if not f.empty])
        elif isinstance(factor , pd.Series):
            factor = factor.to_frame()
            
        assert isinstance(factor , (pd.DataFrame , DataBlock)) , f'factor must be a pandas DataFrame or DataBlock , but got {type(factor)} : {factor}'

        self._df  : pd.DataFrame | Any = None
        self._blk : DataBlock | Any = None
        self._alpha_models : dict[str,AlphaModel] = {}
        self.normalized = normalized
        self.subsets : dict[str,StockFactor] = {}
        self.stats : dict[str,tuple[dict[str,Any],pd.DataFrame]] = {}

        if isinstance(factor , pd.DataFrame):
            if factor.empty:
                factor = pd.DataFrame(columns=['date' , 'secid' , *factor_names])
            factor = factor.reset_index().drop(columns=['index'] , errors='ignore')
            if 'date' in factor.columns: 
                factor = factor.set_index('date' , append=True)
            if 'secid' in factor.columns: 
                factor = factor.set_index('secid' , append=True)
            if None in factor.index.names:
                factor = factor.reset_index([None] , drop=True)
            self._df = factor
        elif isinstance(factor , DataBlock):
            if factor.empty:
                factor = factor.align_feature(factor_names , inplace = True)
            self._blk = factor
        else:
            raise TypeError(f'Unknown factor type: {type(factor)}')

        return self

    def copy(self): 
        """
        return a copy of the factor
        """
        return deepcopy(self)

    def filter_dates(self , dates : np.ndarray | Any | None = None , exclude = False , inplace = False):
        """
        filter the factor data by dates or other index
        """
        if dates is None:
            return self
        df = self._df.query('date not in @dates' if exclude else 'date in @dates')
        return self.update(df , normalized = self.normalized) if inplace else StockFactor(df , normalized = self.normalized)

    def filter_dates_between(self , start_dt : int , end_dt : int , inplace = False):
        """
        filter the factor data by dates between start_dt and end_dt
        """
        dates = self.date
        dates = dates[(dates >= start_dt) & (dates <= end_dt)]
        return self.filter_dates(dates , inplace = inplace)

    def filter_secid(self , secid : np.ndarray | Any | None = None , exclude = False , inplace = False):
        if secid is None: 
            return self
        df = self._df.query('secid not in @secid' if exclude else 'secid in @secid')
        return self.update(df , normalized = self.normalized) if inplace else StockFactor(df , normalized = self.normalized)

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
        if self._df is not None: 
            self._df.rename(columns=mapping , inplace=True)
        if self._blk is not None: 
            self._blk.rename_feature(mapping)
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
        if self._df is None:
            assert self._blk is not None , '_df and _blk cannot be both None'
            self._df = self._blk.to_dataframe()
        return self._df
    
    def block(self) -> DataBlock:
        """
        return the factor data as a DataBlock
        """
        if self._blk is None:
            assert self._df is not None , '_df and _blk cannot be both None'
            self._blk = DataBlock.from_dataframe(self._df)
        return self._blk

    @property
    def secid(self) -> np.ndarray: 
        """
        return the unique secid of the factor
        """
        if self._blk is not None: 
            return self._blk.secid
        else:
            return self._df.index.get_level_values('secid').unique().to_numpy()

    @property
    def date(self) -> np.ndarray:
        """
        return the unique date of the factor
        """
        if self._blk is not None: 
            return self._blk.date
        elif self._df.empty:
            return np.array([])
        else:
            return self._df.index.get_level_values('date').unique().to_numpy()
        
    @property
    def factor_names(self) -> np.ndarray:
        """
        return the factor names of the factor
        """
        if self._blk is not None:
            return self._blk.feature
        else:
            return self._df.columns.to_numpy()
        
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
        if self._df is not None:
            return self._df
        else:
            return self._blk
        
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
        df = DB.load_multi('factor' , factor_name , dates = dates , start_dt=start , end_dt=end)
        return cls(df)
        
    def select(self , secid = None , date = None , factor_name = None):
        """
        select the factor data by secid , date , factor name
        """
        if self._df is not None:
            df = self._df.reset_index()
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
            return StockFactor(self._blk.align(secid , date , factor_name))

    def within(self , benchmark : Benchmark | str | None , recalculate = False) -> 'StockFactor':
        """
        use benchmark to mask factor , only keep the factors that are in the benchmark
        """
        if isinstance(benchmark , str): 
            benchmark = Benchmark(benchmark)
        if not benchmark: 
            return self
        if benchmark.name not in self.subsets or recalculate:
            self.subsets[benchmark.name] = StockFactor(benchmark(self.prior_input) if benchmark else self.prior_input)
        return self.subsets[benchmark.name]
    
    def alpha_model(self , use_cache = True) -> AlphaModel:
        """
        transform the factor to alpha model , only one factor is supported
        """
        assert len(self.factor_names) == 1 , f'only one factor is supported for alpha model , but got {len(self.factor_names)}'
        name = self.factor_names[0]
        if name not in self._alpha_models or not use_cache:
            self._alpha_models[name] = self._get_alpha_model(name)
        return self._alpha_models[name]

    def alpha_models(self , use_cache = True) -> list[AlphaModel]:
        """
        transform the factor to alpha models , multiple factors are supported
        """
        models = []
        for name in self.factor_names:
            if name not in self._alpha_models or not use_cache:
                self._alpha_models[name] = self._get_alpha_model(name)
            models.append(self._alpha_models[name])
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

    def day_returns(self , load = True):
        """
        get the daily quotes for the factor
        """
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
                        nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        """
        return the factor data with additional columns
        """
        df = self.frame()
        if indus:   
            df = append_indus(df)
        if fut_ret: 
            df = append_fut_ret(df , nday , lag , ret_type)
        if ffmv:    
            df = append_ffmv(df)
        return df   

    def eval_ic(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        """
        evaluate the IC of the factor
        """
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}

        if 'ic' not in self.stats or not param_match(self.stats['ic'][0] , params):
            df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
            grouped = df.groupby(by=['date'], as_index=True)
            def df_ic(subdf : pd.DataFrame , **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='An input array is constant; the correlation coefficient is not defined' , category=RuntimeWarning)
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide' , category=RuntimeWarning)
                    return subdf[self.factor_names].corrwith(subdf['ret'], method=ic_type)
            ic = grouped.apply(df_ic , include_groups = False).rename_axis('factor_name',axis='columns')
            self.stats['ic'] = (params , ic)
        return self.stats['ic'][1]
    
    def eval_ic_indus(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                      ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        """
        evaluate the IC of the factor by industry
        """
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        if 'ic_indus' not in self.stats or not param_match(self.stats['ic_indus'][0] , params):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*correlation coefficient is not defined.*')
                df = self.frame_with_cols(indus = True , fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
                def df_ic(subdf : pd.DataFrame , **kwargs):
                    return subdf[self.factor_names].corrwith(subdf['ret'], method=ic_type)
                ic = df.groupby(['date' , 'industry']).apply(df_ic , include_groups = False).\
                    rename_axis('factor_name',axis='columns').reset_index(drop=False).\
                    melt(id_vars = ['date' , 'industry'] , var_name = 'factor_name' , value_name = 'ic_indus')
            self.stats['ic_indus'] = (params , ic)
        return self.stats['ic_indus'][1]
    
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
                        ret_type : Literal['close' , 'vwap'] = 'close' , trade_date = True) -> pd.DataFrame:
        """
        evaluate the group performance of the factor
        """
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'excess' : excess , 'ret_type' : ret_type , 'trade_date' : trade_date}
        if 'group_perf' not in self.stats or not param_match(self.stats['group_perf'][0] , params):
            df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
            # assert not np.any(np.isinf(df['ret'])), f'inf values in factor data ret : {df[np.isinf(df).any(axis=1)]}'
            df = self._eval_group_perf(df , self.factor_names , group_num , excess)
            if trade_date:
                df['start'] = DATAVENDOR.td_array(df['date'] , lag)
                df['end']   = DATAVENDOR.td_array(df['date'] , lag + nday - 1)
            self.stats['group_perf'] = (params , df)
        return self.stats['group_perf'][1]
    
    def eval_weighted_pnl(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                          ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                          weight_type_list : list[str] = ['long' , 'long_short' , 'short']) -> pd.DataFrame:
        """
        evaluate the weighted pnl of the factor
        """
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type , 'given_direction' : given_direction , 'weight_type_list' : weight_type_list}
        if 'weighted_pnl' not in self.stats or not param_match(self.stats['weighted_pnl'][0] , params):
            df = self.frame_with_cols(fut_ret=True , nday=nday , lag=lag , ret_type=ret_type)
            direction = given_direction if given_direction else np.sign(df.corr().loc['ret'].drop('ret')) 

            dfs = []
            for wt in weight_type_list:
                kwargs = {'weight_type' : wt , 'direction' : direction , 'group_num' : group_num , 'include_groups' : False}
                df_wt = df.groupby('date').apply(eval_weighted_pnl, **kwargs).\
                    reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ret').assign(weight_type = wt)
                dfs.append(df_wt)

            pnl = pd.concat(dfs, axis=0).set_index(['weight_type' , 'date'])
            pnl = pnl.join(pnl.groupby(['factor_name' , 'weight_type']).cumsum().rename(columns={'ret':'cum_ret'})).reset_index()
            self.stats['weighted_pnl'] = (params , pnl)
        return self.stats['weighted_pnl'][1]

    def coverage(self , benchmark : Benchmark | str | None = None):
        """
        evaluate the coverage of the factor by benchmark
        """
        params = {'benchmark' : benchmark.name if isinstance(benchmark,Benchmark) else benchmark}
        if 'coverage' not in self.stats or not param_match(self.stats['coverage'][0] , params):
            dates = self.date
            if isinstance(benchmark , str) or benchmark is None: 
                benchmark = Benchmark(benchmark)
            factor = self.within(benchmark)
            benchmark_size = pd.Series(benchmark.sec_num(dates) , index = dates)
            coverage = factor.frame().groupby('date').apply(lambda x:x.dropna().count(numeric_only=True))
            for factor_name in coverage:
                coverage[factor_name] = (coverage[factor_name] / benchmark_size).clip(lower=0 , upper=1)
            self.stats['coverage'] = (params , coverage)
        return self.stats['coverage'][1]

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
            return StockFactor(df , normalized = True)
    
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