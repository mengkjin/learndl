import numpy as np
import pandas as pd

from collections.abc import Iterable
from copy import deepcopy
from typing import Any , Literal , Optional , Union
from src.data import DataBlock , DATAVENDOR
from src.func import transform as T

from .alpha_model import AlphaModel
from .benchmark import Benchmark

__all__ = ['StockFactor']

def append_indus(df : pd.DataFrame):
    secid = df.index.get_level_values('secid').unique().values
    date = df.index.get_level_values('date').unique().values
    old_index = df.index.names
    appending = DATAVENDOR.risk_industry_exp(secid , date).to_dataframe().fillna('unknown')
    appending = pd.DataFrame(appending.idxmax(axis=1).rename('industry') , index = appending.index)
    df = df.reset_index().merge(appending , on = ['secid','date']).set_index(old_index)
    return df

def append_ffmv(df : pd.DataFrame):
    secid = df.index.get_level_values('secid').unique().values
    date = df.index.get_level_values('date').unique().values
    old_index = df.index.names
    appending = DATAVENDOR.ffmv(secid , date).to_dataframe()
    df = df.reset_index().merge(appending , on = ['secid','date']).set_index(old_index)
    return df

def append_fut_ret(df : pd.DataFrame , nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close'):
    secid = df.index.get_level_values('secid').unique().values
    date = df.index.get_level_values('date').unique().values
    old_index = df.index.names
    appending = DATAVENDOR.nday_fut_ret(secid , date , nday , lag , ret_type = ret_type).to_dataframe()
    df = df.reset_index().merge(appending , on = ['secid','date']).set_index(old_index)
    return df

def melt_frame(df : pd.DataFrame):
    if 'factor_name' in df.index.names:
        return df
    else:
        return df.melt(var_name = 'factor_name' , ignore_index = False).set_index('factor_name' , append=True)

def pivot_frame(df : pd.DataFrame):
    if 'factor_name' in df.index.names:
        return df.pivot_table(index = ['date' , 'secid'] , columns = 'factor_name' , values = 'value')
    else:
        return df

def whiten(df : pd.DataFrame | Any, ffmv_weighted = False , pivot = True):
    df = melt_frame(df)
    if ffmv_weighted:
        df = append_ffmv(df)
        # df = df.groupby(by=['date' , 'factor_name'] , group_keys=False).apply(lambda x:whiten(x['value'] , weight = x['weight']))
        df = df.groupby(by=['date' , 'factor_name']).transform(lambda x:T.whiten(x['value'] , weight = x['weight']))
    else:
        df = df.groupby(by=['date' , 'factor_name']).transform(T.whiten)
    # if isinstance(df , pd.Series): df = df.to_frame()
    if pivot: df = pivot_frame(df)
    return df

def winsor(df : pd.DataFrame | Any , pivot = True , **kwargs):
    df = melt_frame(df)
    df = df.groupby(by=['date' , 'factor_name']).transform(T.winsorize , **kwargs)
    if isinstance(df , pd.Series): df = df.to_frame()
    if pivot: df = pivot_frame(df)
    return df

def fillna(df : pd.DataFrame | Any , 
           fill_method : Literal['zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'zero' ,
           pivot = True , **kwargs):
    df = melt_frame(df)
    if fill_method == 'zero':
        ...
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
    if isinstance(df , pd.Series): df = df.to_frame()
    if pivot: df = pivot_frame(df)
    return df

def neutralize(df : pd.DataFrame | Any , pivot = True , **kwargs):
    if pivot: df = pivot_frame(df)
    return df

def eval_grp_avg(x : pd.DataFrame , x_cols : list, y_name : str = 'ret', group_num : int = 10 , excess = False) -> pd.DataFrame:
    y = pd.DataFrame(x[y_name], index=x.index, columns=[y_name])
    rtn = list()
    for col in x_cols:
        bins = x[col].drop_duplicates().quantile(np.linspace(0,1,group_num + 1))
        y['group'] = pd.cut(x[col], bins=bins, labels=[i for i in range(1, group_num + 1)])
        if excess: y[y_name] -= y[y_name].mean()
        grp_avg_ret = y.groupby('group' , observed = True)[y_name].mean().rename(col)
        rtn.append(grp_avg_ret)
    rtn = pd.concat(rtn, axis=1, sort=True)
    return rtn

def pnl_weights(x : pd.DataFrame, weight_type : str, direction : Any = 1 , group_num : int = 10):
    assert weight_type in ['long_short', 'top100' , 'long', 'short'] , weight_type
    assert np.all(np.sign(direction) != 0) , direction
    x = (x / np.nanstd(x, axis=0)) * np.sign(direction)
    norm_weight = lambda xx : (xx / np.sum(xx, axis=0))
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
    vals = x[x.columns.drop([y_name])]
    rets = x[[y_name]].to_numpy()
    weights = pnl_weights(vals, weight_type, direction, group_num)
    rtn = (weights * rets).sum(axis = 0)
    return rtn

class StockFactor:
    def __init__(self , factor : Union[pd.DataFrame,pd.Series,DataBlock,'StockFactor'] , normalized = False):
        self.update(factor , normalized)

    def __repr__(self):
        return f'{self.__class__.__name__}(normalized={self.normalized},names={self.factor_names},dates={self.date.min()}-{self.date.max()})'
    
    def __call__(self , benchmark):
        return self.within(benchmark)

    def update(self , factor : Union[pd.DataFrame,pd.Series,DataBlock,'StockFactor'] , normalized = False):
        if isinstance(factor , StockFactor):
            factor = factor.prior_input

        self._df  : pd.DataFrame | Any = None
        self._blk : DataBlock | Any = None
        if isinstance(factor , pd.Series): factor = factor.to_frame()
        if isinstance(factor , pd.DataFrame):
            if 'date' not in factor.index.names: factor = factor.set_index('date' , append=True)
            if 'secid' not in factor.index.names: factor = factor.set_index('secid' , append=True)
            if None in factor.index.names: factor = factor.reset_index([None] , drop=True)
            self._df = factor
        else:
            self._blk = factor
        self.normalized = normalized
        self.subsets : dict[str,StockFactor] = {}
        return self

    def copy(self): return deepcopy(self)

    def frame(self) -> pd.DataFrame:
        if self._df is None:
            assert self._blk is not None , '_df and _blk cannot be both None'
            self._df = self._blk.to_dataframe()
        return self._df
    
    def block(self) -> DataBlock:
        if self._blk is None:
            assert self._df is not None , '_df and _blk cannot be both None'
            self._blk = DataBlock.from_dataframe(self._df)
        return self._blk

    @property
    def secid(self) -> np.ndarray: 
        if self._blk is not None: 
            return self._blk.secid
        else:
            return self._df.index.get_level_values('secid').unique().values

    @property
    def date(self) -> np.ndarray:
        if self._blk is not None: 
            return self._blk.date
        else:
            return self._df.index.get_level_values('date').unique().values
        
    @property
    def factor_names(self) -> np.ndarray:
        if self._blk is not None:
            return self._blk.feature
        else:
            return self._df.columns.to_numpy()

    @property
    def prior_input(self) -> pd.DataFrame | DataBlock:
        if self._df is not None:
            return self._df
        else:
            return self._blk
        
    def select(self , secid = None , date = None , factor_name = None):
        if self._df is not None:
            df = self._df.reset_index()
            if date is not None:  df = df[df['date'].isin(date if isinstance(date , Iterable) else [date])]
            if secid is not None: df = df[df['secid'].isin(secid if isinstance(secid , Iterable) else [secid])]
            df = df.set_index(['date' , 'secid'])
            if factor_name is not None: df = df[factor_name]
            return StockFactor(df)
        else:
            return StockFactor(self._blk.align(secid , date , factor_name , inplace = False))

    def within(self , benchmark : Optional[Benchmark | str] , recalculate = False) -> 'StockFactor':
        '''use benchmark to mask factor'''
        if isinstance(benchmark , str): benchmark = Benchmark(benchmark)
        if not benchmark: return self
        if benchmark.name not in self.subsets or recalculate:
            self.subsets[benchmark.name] = StockFactor(benchmark(self.prior_input) if benchmark else self.prior_input)
        return self.subsets[benchmark.name]
    
    def alpha_model(self) -> AlphaModel:
        assert len(self.factor_names) == 1 , f'only one factor is supported for alpha model , but got {len(self.factor_names)}'
        return self.alpha_models()[0]

    def alpha_models(self) -> list[AlphaModel]:
        return [AlphaModel.from_dataframe(data , col) for col , data in self.frame().items()]

    def frame_with_cols(self , indus = False , fut_ret = False , ffmv = False ,
                        nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close'):
        df = self.frame()
        if indus:   df = append_indus(df)
        if fut_ret: df = append_fut_ret(df , nday , lag , ret_type)
        if ffmv:    df = append_ffmv(df)
        return df   

    def eval_ic(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        factors = self.factor_names
        df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
        ic = df.groupby(by=['date'], as_index=True).apply(lambda x:x[factors].corrwith(x['ret'], method=ic_type))
        return ic.rename_axis('factor_name',axis='columns')
    
    def eval_ic_indus(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                      ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        with np.errstate(all='ignore'):
            df = self.frame_with_cols(indus = True , fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
            ic = df.groupby(['date' , 'industry']).apply(lambda x:x[self.factor_names].corrwith(x['ret'], method=ic_type) , include_groups=False)

        return ic.rename_axis('factor_name',axis='columns')
    
    def eval_group_perf(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 , excess = False , 
                        ret_type : Literal['close' , 'vwap'] = 'close' , trade_date = True) -> pd.DataFrame:
        df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
        kwargs = {'x_cols' : self.factor_names , 'y_name' : 'ret' , 'group_num' : group_num , 'excess' : excess}
        df = df.groupby('date').apply(lambda x:eval_grp_avg(x , **kwargs)) 
        df = df.melt(var_name='factor_name' , value_name='group_ret' , ignore_index=False).sort_values(['date','factor_name']).reset_index()
        if trade_date:
            df['start'] = DATAVENDOR.td_array(df['date'] , lag)
            df['end']   = DATAVENDOR.td_array(df['date'] , lag + nday - 1)
        return df
    
    def eval_weighted_pnl(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                          ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                          weight_type_list : list[str] = ['long' , 'long_short' , 'short']) -> pd.DataFrame:
        df = self.frame_with_cols(fut_ret=True , nday=nday , lag=lag , ret_type=ret_type)
        direction = given_direction if given_direction else np.sign(df.corr().loc['ret'].drop('ret')) 

        dfs = []
        for wt in weight_type_list:
            df_wt = df.groupby('date').apply(lambda x:eval_weighted_pnl(x, wt, direction, group_num)).\
                reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ret').assign(weight_type = wt)
            dfs.append(df_wt)

        pnl = pd.concat(dfs, axis=0).set_index(['weight_type' , 'date'])
        pnl = pnl.join(pnl.groupby(['factor_name' , 'weight_type']).cumsum().rename(columns={'ret':'cum_ret'})).reset_index()
        return pnl
    
    def normalize(self , fill_method : Literal['zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'zero' ,
                  weighted_whiten = False , order = ['fillna' , 'whiten' , 'winsor'] , inplace = False):
        df = self.frame()
        for step in order:
            if step == 'fillna':   df = fillna(df , fill_method = fill_method)
            elif step == 'whiten': df = whiten(df , ffmv_weighted = weighted_whiten)
            elif step == 'winsor': df = winsor(df)
        df = pivot_frame(df)
        if inplace: 
            self.update(df)
            return self
        else:
            return StockFactor(df , normalized = True)
    
    