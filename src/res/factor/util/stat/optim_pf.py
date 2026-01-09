import pandas as pd

from typing import Any , Literal

from .basic import eval_pf_stats , eval_cum_ret , eval_drawdown
from ..agency import BaseConditioner

def _filter_account(acc : pd.DataFrame , lag0 = True , pos_model_date = False):
    '''drop lag if exists , and select lag0'''
    if lag0:
        if 'lag' in acc.index.names: 
            acc = acc.reset_index('lag',drop=False)
        if 'lag' in acc.columns: 
            acc = acc.query('lag==0').drop(columns=['lag'])
    if pos_model_date:
        acc = acc.query('model_date>0')
    return acc

def calc_frontface(acc : pd.DataFrame):
    grouped = acc.groupby(acc.index.names , observed=True)
    start : pd.Series | Any = grouped['start'].min()
    end : pd.Series | Any = grouped['end'].max()
    basic = pd.concat([start , end] , axis=1)
    stats = grouped.apply(eval_pf_stats , include_groups=False).reset_index([None],drop=True)
    df = basic.join(stats).sort_index()
    return df

def calc_perf_curve(acc : pd.DataFrame):
    df = _filter_account(acc).loc[:,['end','pf','bm','excess']]
    index_names = [str(name) for name in df.index.names]
    df = df.sort_values([*index_names , 'end']).rename(columns={'end':'trade_date'})

    df[['bm','pf']] = eval_cum_ret(df[['bm','pf']] , 'exp' , groupby=index_names)
    df['excess'] = eval_cum_ret(df['excess'] , 'lin' , groupby=index_names)
    df = df.set_index('trade_date' , append=True)
    return df

def calc_perf_drawdown(acc : pd.DataFrame):
    df = _filter_account(acc).loc[:,['end','pf','overnight']]
    index_names = [str(name) for name in df.index.names]
    df = df.sort_values([*index_names , 'end']).rename(columns={'end':'trade_date'})
    df['drawdown'] = eval_drawdown(df['pf'] , 'exp' , groupby = index_names)
    conditioners = [BaseConditioner.select_conditioner(name)() 
                    for name in ['balance' , 'conservative' , 'radical']]  
    dfs = [] 
    for grp , sub in df.groupby(df.index.names , observed=True):
        sub_index_names = [str(name) for name in sub.index.names]
        sub['raw'] = eval_cum_ret(sub['pf'] , 'exp' , groupby = sub_index_names)
        for conditioner in conditioners:
            sub[conditioner.conditioner_name()] = conditioner.conditioned_pf_ret(sub['pf'] , plot = False)
        sub = sub.set_index('trade_date' , append=True).drop(columns=['pf','overnight'])
        dfs.append(sub)
    df = pd.concat(dfs)

    return df

def calc_perf_excess_drawdown(acc : pd.DataFrame):
    df = _filter_account(acc).loc[:,['end','excess']]
    index_names = [str(name) for name in df.index.names]
    df = df.sort_values([*index_names , 'end']).rename(columns={'end':'trade_date'})
    df['excess'] = eval_cum_ret(df['excess'] , 'lin' , groupby=index_names)
    df['peak']   = df.groupby(index_names , observed=True)[['excess']].cummax()
    df['drawdown'] = df['excess'] - df['peak']
    df = df.set_index('trade_date' , append=True).loc[:,['excess' , 'drawdown']]
    return df

def calc_perf_lag(acc : pd.DataFrame):
    if 'lag' not in acc.index.names and 'lag' not in acc.columns: 
        return pd.DataFrame()
    df = acc.loc[:,['end','excess']].copy()
    index_names = [str(name) for name in df.index.names]
    df = df.sort_values([*index_names , 'end']).rename(columns={'end':'trade_date'})
    df = df.set_index('trade_date',append=True).groupby(df.index.names , observed=True)[['excess']].cumsum()
    if 'suffix' in df.index.names: 
        df = df.reset_index('suffix' , drop=True)
    df = df.pivot_table('excess',[f for f in df.index.names if f != 'lag'],'lag', observed=True)
    lag_max = max(df.columns.values)
    lag_min = min(df.columns.values)
    df.columns = [f'lag{col}' for col in df.columns]
    df['lag_cost'] = df[f'lag{lag_min}'] - df[f'lag{lag_max}']
    return df

def calc_perf_period(acc : pd.DataFrame , period : Literal['year' , 'yearmonth' , 'month'] = 'year'):
    '''Calculate performance stats for each period'''
    if period=='year': 
        acc[period] = acc['end'].astype(str).str[:4]
    elif period == 'yearmonth':  
        acc[period] = acc['end'].astype(str).str[:6]
    else: 
        acc[period] = acc['end'].astype(str).str[4:6]
        
    acc = _filter_account(acc)
    # calculate period stats and all stats, then concat them
    apply_kwargs = {'mdd_period': (period != 'month') , 'include_groups': False}
    prd_stat = acc.groupby([*acc.index.names , period] , observed=True).apply(eval_pf_stats , **apply_kwargs).reset_index(period)
    all_stat = acc.groupby(acc.index.names , observed=True).apply(eval_pf_stats , **apply_kwargs).assign(**{period:'ALL'})
    df = pd.concat([prd_stat , all_stat]).reset_index([None],drop=True)
    return df

def calc_perf_year(acc : pd.DataFrame):
    '''Calculate performance stats for each year'''
    return calc_perf_period(acc , 'year')

def calc_perf_month(acc : pd.DataFrame):
    '''Calculate performance stats for each calendar month'''
    return calc_perf_period(acc , 'month')

def calc_exp_style(acc : pd.DataFrame):
    def fetch_exp_style(x : pd.Series , **kwargs) -> pd.DataFrame:
        return x.iloc[0].style.loc[:,['active']]
    df = _filter_account(acc , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_style , include_groups = False).reset_index('style')
    df = df.pivot_table('active' , df.index.names , columns='style' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_exp_indus(acc : pd.DataFrame):
    def fetch_exp_indus(x : pd.Series , **kwargs) -> pd.DataFrame:
        return x.iloc[0].industry.loc[:,['active']]
    df = _filter_account(acc , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_indus , include_groups = False).reset_index('industry')
    df = df.pivot_table('active' , df.index.names , columns='industry' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_attrib_source(acc : pd.DataFrame):
    def fetch_attrib_source(x : pd.Series , **kwargs) -> pd.DataFrame:
        return x.iloc[0].source.loc[:,['contribution']].rename_axis('source')
    df = _filter_account(acc , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_source , include_groups = False).reset_index('source')
    df = df.pivot_table('contribution' , df.index.names , columns='source' , observed=True).\
        loc[:,['tot' , 'excess' , 'market' , 'industry' , 'style' , 'specific' , 'cost']].\
        rename_axis(None , axis='columns').sort_index().\
        groupby([col for col in df.index.names if col != 'end'] , observed=True).cumsum()
    df = df.rename_axis(index={'end':'trade_date'})
    return df

def calc_attrib_style(acc : pd.DataFrame):
    def fetch_attrib_style(x : pd.Series , **kwargs) -> pd.DataFrame:
        return x.iloc[0].style.loc[:,['contribution']].rename_axis('style')
    df = _filter_account(acc , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_style , include_groups = False).reset_index('style')
    if isinstance(df , pd.Series): 
        df = df.to_frame()
    df = df.pivot_table('contribution' , df.index.names , columns='style' , observed=True).\
        rename_axis(None , axis='columns').sort_index().\
        groupby([col for col in df.index.names if col != 'end'] , observed=True).cumsum()
    df = df.rename_axis(index={'end':'trade_date'})
    return df
