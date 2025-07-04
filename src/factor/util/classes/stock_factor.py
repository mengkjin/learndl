import numpy as np
import pandas as pd

from collections.abc import Iterable
from copy import deepcopy
from os import listdir
from typing import Any , Literal , Optional , Union

from src.basic import INSTANCE_RECORD , PATH
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
           fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'zero' ,
           pivot = True , **kwargs):
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

def param_match(args1 : dict[str,Any] , args2 : dict[str,Any]) -> bool:
    if len(args1) != len(args2): return False
    matches = [k in args2 and args2[k] == v for k , v in args1.items()]
    return all(matches)

class StockFactor:
    def __init__(self , factor : Union[pd.DataFrame,pd.Series,DataBlock,'StockFactor',dict[int,pd.Series]] , normalized = False , step = None):
        self.update(factor , normalized , step)
        INSTANCE_RECORD.update_factor(self)

    def __repr__(self):
        return f'{self.__class__.__name__}(normalized={self.normalized},names={self.factor_names},dates={self.date.min()}-{self.date.max()})'
    
    def __call__(self , benchmark):
        return self.within(benchmark)

    def update(self , factor : Union[pd.DataFrame,pd.Series,DataBlock,'StockFactor',dict[int,pd.Series]] , normalized = False , step = None):
        if isinstance(factor , StockFactor):
            factor = factor.prior_input
        elif isinstance(factor , dict):
            factor = pd.concat([f.to_frame().assign(date = d) for d , f in factor.items() if not f.empty])
        elif isinstance(factor , pd.Series):
            factor = factor.to_frame()

        self._df  : pd.DataFrame | Any = None
        self._blk : DataBlock | Any = None

        if isinstance(factor , pd.DataFrame):
            if 'date' not in factor.index.names: factor = factor.set_index('date' , append=True)
            if 'secid' not in factor.index.names: factor = factor.set_index('secid' , append=True)
            if None in factor.index.names: factor = factor.reset_index([None] , drop=True)
            self._df = factor
        else:
            self._blk = factor

        self.normalized = normalized
        self.subsets : dict[str,StockFactor] = {}
        self.step = step
        self.stats : dict[str,tuple[dict[str,Any],pd.DataFrame]] = {}
        return self

    def copy(self): return deepcopy(self)

    def rename(self , new_name : str | list[str] | dict[str,str] , inplace = True):
        if not inplace: self = self.copy()
        if isinstance(new_name , str):
            assert self.factor_num == 1 , f'only one factor is supported for using str for renaming : {self.factor_names}'
            mapping = {self.factor_names[0]:new_name}
        elif isinstance(new_name , list):
            assert len(new_name) == self.factor_num , f'the length of new_name must be equal to the number of factors : {self.factor_names}'
            mapping = {self.factor_names[i]:new_name[i] for i in range(len(new_name))}
        elif isinstance(new_name , dict):
            mapping = new_name
        if self._df is not None: self._df.rename(columns=mapping , inplace=True)
        if self._blk is not None: self._blk.rename_feature(mapping)
        return self

    def join(self , *others : 'StockFactor'):
        df = self.frame()
        for other in others: df = df.join(other.frame() , how = 'outer')
        return StockFactor(df)
    
    def ew(self):
        if self.factor_num == 1:
            return self
        else:
            df = self.frame().mean(axis = 1).rename('multifactor_ew')
            return StockFactor(df)

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
    def factor_num(self) -> int:
        return self.prior_input.shape[-1]

    @property
    def prior_input(self) -> pd.DataFrame | DataBlock:
        if self._df is not None:
            return self._df
        else:
            return self._blk
        
    @classmethod
    def Load(cls , factor_name : str , date : int):
        df = PATH.factor_load(factor_name , date).assign(date = date)
        return cls(df)

    @classmethod
    def Loads(cls , factor_name : str , start : int | None = None , end : int | None = None):
        if start is None: start = 0
        if end is None:   end = 99991231
        df = PATH.factor_load_multi(factor_name , start_dt=start , end_dt=end)
        return cls(df)
        
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

    def eval_ic(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        if 'ic' not in self.stats or not param_match(self.stats['ic'][0] , params):
            factors = self.factor_names
            df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
            ic = df.groupby(by=['date'], as_index=True).apply(
                lambda x:x[factors].corrwith(x['ret'], method=ic_type)).\
                rename_axis('factor_name',axis='columns')
            self.stats['ic'] = (params , ic)
        return self.stats['ic'][1]
    
    def eval_ic_indus(self , nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'spearman' ,
                      ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        params = {'nday' : nday , 'lag' : lag , 'ic_type' : ic_type , 'ret_type' : ret_type}
        if 'ic_indus' not in self.stats or not param_match(self.stats['ic_indus'][0] , params):
            with np.errstate(all='ignore'):
                df = self.frame_with_cols(indus = True , fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
                ic = df.groupby(['date' , 'industry']).apply(
                    lambda x:x[self.factor_names].corrwith(x['ret'], method=ic_type) , include_groups=False).\
                    rename_axis('factor_name',axis='columns')
            self.stats['ic_indus'] = (params , ic)
        return self.stats['ic_indus'][1]
    
    def eval_group_perf(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 , excess = False , 
                        ret_type : Literal['close' , 'vwap'] = 'close' , trade_date = True) -> pd.DataFrame:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'excess' : excess , 'ret_type' : ret_type , 'trade_date' : trade_date}
        if 'group_perf' not in self.stats or not param_match(self.stats['group_perf'][0] , params):
            df = self.frame_with_cols(fut_ret = True , nday = nday , lag = lag , ret_type = ret_type)
            kwargs = {'x_cols' : self.factor_names , 'y_name' : 'ret' , 'group_num' : group_num , 'excess' : excess}
            df = df.groupby('date').apply(lambda x:eval_grp_avg(x , **kwargs)) 
            df = df.melt(var_name='factor_name' , value_name='group_ret' , ignore_index=False).sort_values(['date','factor_name']).reset_index()
            if trade_date:
                df['start'] = DATAVENDOR.td_array(df['date'] , lag)
                df['end']   = DATAVENDOR.td_array(df['date'] , lag + nday - 1)
            self.stats['group_perf'] = (params , df)
        return self.stats['group_perf'][1]
    
    def eval_weighted_pnl(self , nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                          ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                          weight_type_list : list[str] = ['long' , 'long_short' , 'short']) -> pd.DataFrame:
        params = {'nday' : nday , 'lag' : lag , 'group_num' : group_num , 'ret_type' : ret_type , 'given_direction' : given_direction , 'weight_type_list' : weight_type_list}
        if 'weighted_pnl' not in self.stats or not param_match(self.stats['weighted_pnl'][0] , params):
            df = self.frame_with_cols(fut_ret=True , nday=nday , lag=lag , ret_type=ret_type)
            direction = given_direction if given_direction else np.sign(df.corr().loc['ret'].drop('ret')) 

            dfs = []
            for wt in weight_type_list:
                df_wt = df.groupby('date').apply(lambda x:eval_weighted_pnl(x, wt, direction, group_num)).\
                    reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ret').assign(weight_type = wt)
                dfs.append(df_wt)

            pnl = pd.concat(dfs, axis=0).set_index(['weight_type' , 'date'])
            pnl = pnl.join(pnl.groupby(['factor_name' , 'weight_type']).cumsum().rename(columns={'ret':'cum_ret'})).reset_index()
            self.stats['weighted_pnl'] = (params , pnl)
        return self.stats['weighted_pnl'][1]
    
    def coverage(self , benchmark : Optional[Benchmark | str] = None):
        params = {'benchmark' : benchmark.name if isinstance(benchmark,Benchmark) else benchmark}
        if 'coverage' not in self.stats or not param_match(self.stats['coverage'][0] , params):
            dates = self.date
            if isinstance(benchmark , str) or benchmark is None: benchmark = Benchmark(benchmark)
            factor = self.within(benchmark)
            benchmark_size = pd.Series(benchmark.sec_num(dates) , index = dates)
            coverage = factor.frame().groupby('date').apply(lambda x:x.dropna().count(numeric_only=True))
            for factor_name in coverage:
                coverage[factor_name] = (coverage[factor_name] / benchmark_size).clip(lower=0 , upper=1)
            self.stats['coverage'] = (params , coverage)
        return self.stats['coverage'][1]
    
    def normalize(self , fill_method : Literal['drop' , 'zero' ,'ffill' , 'mean' , 'median' , 'indus_mean' , 'indus_median'] = 'drop' ,
                  weighted_whiten = False , order = ['fillna' , 'winsor' , 'whiten'] , inplace = False):
        df = self.frame()
        for step in order:
            if step == 'fillna':   df = fillna(df , fill_method = fill_method)
            elif step == 'whiten': df = whiten(df , ffmv_weighted = weighted_whiten)
            elif step == 'winsor': df = winsor(df)
        df = pivot_frame(df)
        if inplace: 
            return self.update(df , normalized = True)
        else:
            return StockFactor(df , normalized = True)
    
    def select_analytic(self , task_name : str , **kwargs):
        from src.factor.analytic import FactorPerfManager , BasePerfCalc

        match_task = [task for task in FactorPerfManager.TASK_LIST if task.match_name(task_name)]
        assert match_task and len(match_task) <= 1 , f'no match or duplicate match tasks : {task_name}'
        task , task_name = match_task[0] , match_task[0].__name__
        if not hasattr(self , 'analytic_tasks'): self.analytic_tasks : dict[str , BasePerfCalc] = {}
        if task_name not in self.analytic_tasks: self.analytic_tasks[task_name] = task(**kwargs)
        return self.analytic_tasks[task_name]

    def analyze(self , 
                task_name : Literal['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Decay', 'IC_Indus',
                                    'IC_Year','IC_Benchmark','IC_Monotony','PnL_Curve',
                                    'Style_Corr','Group_Curve','Group_Decay','Group_IR_Decay',
                                    'Group_Year','Distrib_Curve'] | str , plot = True , display = True , **kwargs):
        if self.step is not None and 'nday' not in kwargs: kwargs['nday'] = self.step
        task = self.select_analytic(task_name , **kwargs)
        task.calc(self)
        if plot: task.plot(show = display)
        return self

    def fast_analyze(self , task_list = ['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Benchmark','IC_Monotony','Style_Corr'] , **kwargs):
        '''['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Benchmark','IC_Monotony','Style_Corr']'''
        for task_name in task_list: self.analyze(task_name , **kwargs)
        return self

    def full_analyze(self , **kwargs):
        task_list = ['FrontFace', 'Coverage' , 'IC_Curve', 'IC_Decay', 'IC_Indus',
                     'IC_Year','IC_Benchmark','IC_Monotony','PnL_Curve',
                     'Style_Corr','Group_Curve','Group_Decay','Group_IR_Decay',
                     'Group_Year','Distrib_Curve']
        for task_name in task_list: self.analyze(task_name , **kwargs)
        return self