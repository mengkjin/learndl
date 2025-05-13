import pandas as pd
import numpy as np

from typing import Any , Literal

from src.data import DATAVENDOR
from src.factor.util.stat import eval_pf_stats , eval_cum_ret , eval_drawdown
from src.factor.util.agency import BaseConditioner

def filter_account(account : pd.DataFrame , lag0 = True , pos_model_date = False):
    '''drop lag if exists , and select lag0'''
    if lag0:
        if 'lag' in account.index.names: account = account.reset_index('lag',drop=False)
        if 'lag' in account.columns: account = account[account['lag']==0].drop(columns=['lag'])
    if pos_model_date:
        if 'model_date' in account.columns:
            account = account[account['model_date']>0]
        elif 'model_date' in account.index.names:
            account = account[account.index.get_level_values('model_date')>0]
    return account

def calc_optim_frontface(account : pd.DataFrame):
    grouped = account.groupby(account.index.names , observed=True)
    basic = pd.concat([grouped['start'].min() , grouped['end'].max()] , axis=1)
    stats = grouped.apply(eval_pf_stats , include_groups=False).reset_index([None],drop=True)
    df = basic.join(stats).sort_index()
    return df

def calc_optim_perf_curve(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','pf','bm','excess']]
    df = df.sort_values([*df.index.names , 'end']).rename(columns={'end':'trade_date'})

    df[['bm','pf']] = eval_cum_ret(df[['bm','pf']] , 'exp' , groupby=df.index.names)
    df['excess'] = eval_cum_ret(df['excess'] , 'lin' , groupby=df.index.names)
    df = df.set_index('trade_date' , append=True)
    return df

def calc_optim_perf_drawdown(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','pf','overnight']]
    df = df.sort_values([*df.index.names , 'end']).rename(columns={'end':'trade_date'})
    df['drawdown'] = eval_drawdown(df['pf'] , 'exp' , groupby = df.index.names)
    conditioners = [BaseConditioner.select_conditioner(name)() 
                    for name in ['balance' , 'conservative' , 'radical']]  
    dfs = [] 
    for grp , sub in df.groupby(df.index.names , observed=True):
        sub['raw'] = eval_cum_ret(sub['pf'] , 'exp' , groupby = sub.index.names)
        for conditioner in conditioners:
            sub[conditioner.conditioner_name()] = conditioner.conditioned_pf_ret(sub , plot = False)
        sub = sub.set_index('trade_date' , append=True).drop(columns=['pf','overnight'])
        dfs.append(sub)
    df = pd.concat(dfs)

    return df

def calc_optim_perf_excess_drawdown(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','excess']]
    df = df.sort_values([*df.index.names , 'end']).rename(columns={'end':'trade_date'})
    df['excess'] = eval_cum_ret(df['excess'] , 'lin' , groupby=df.index.names)
    df['peak']   = df.groupby(df.index.names , observed=True)[['excess']].cummax()
    df['drawdown'] = df['excess'] - df['peak']
    df = df.set_index('trade_date' , append=True).loc[:,['excess' , 'drawdown']]
    return df

def calc_optim_perf_lag(account : pd.DataFrame):
    if 'lag' not in account.index.names and 'lag' not in account.columns: return pd.DataFrame()
    df = account.loc[:,['end','excess']].copy()
    df = df.sort_values([*df.index.names , 'end']).rename(columns={'end':'trade_date'})
    df = df.groupby(df.index.names , observed=True)[['excess']].cumsum()
    df = df.pivot_table('excess',[f for f in df.index.names if f != 'lag'],'lag', observed=True)
    lag_max = max(df.columns.values)
    lag_min = min(df.columns.values)
    df.columns = [f'lag{col}' for col in df.columns]
    df['lag_cost'] = df[f'lag{lag_min}'] - df[f'lag{lag_max}']
    return df

def calc_optim_perf_period(account : pd.DataFrame , period : Literal['year' , 'yearmonth' , 'month'] = 'year'):
    '''Calculate performance stats for each period'''
    if period=='year': 
        account[period] = account['end'].astype(str).str[:4]
    elif period == 'yearmonth':  
        account[period] = account['end'].astype(str).str[:6]
    else: 
        account[period] = account['end'].astype(str).str[4:6]
        
    account = filter_account(account)
    # calculate period stats and all stats, then concat them
    apply_kwargs = {'mdd_period': (period != 'month') , 'include_groups': False}
    prd_stat = account.groupby([*account.index.names , period] , observed=True).apply(eval_pf_stats , **apply_kwargs).reset_index(period)
    all_stat = account.groupby(account.index.names , observed=True).apply(eval_pf_stats , **apply_kwargs).assign(**{period:'ALL'})
    df = pd.concat([prd_stat , all_stat]).reset_index([None],drop=True)
    return df

def calc_optim_perf_year(account : pd.DataFrame):
    '''Calculate performance stats for each year'''
    return calc_optim_perf_period(account , 'year')

def calc_optim_perf_month(account : pd.DataFrame):
    '''Calculate performance stats for each calendar month'''
    return calc_optim_perf_period(account , 'month')

def fetch_exp_style(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].style.loc[:,['active']]

def fetch_exp_indus(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].industry.loc[:,['active']]

def fetch_attrib_source(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].source.loc[:,['contribution']].rename_axis('source')

def fetch_attrib_style(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].style.loc[:,['contribution']].rename_axis('style')

def calc_optim_exp_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_style).reset_index('style')
    df = df.pivot_table('active' , df.index.names , columns='style' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_optim_exp_indus(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_indus).reset_index('industry')
    df = df.pivot_table('active' , df.index.names , columns='industry' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_optim_attrib_source(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_source).reset_index('source')
    df = df.pivot_table('contribution' , df.index.names , columns='source' , observed=True).\
        loc[:,['tot' , 'excess' , 'market' , 'industry' , 'style' , 'specific' , 'cost']].\
        rename_axis(None , axis='columns').sort_index().\
        groupby([col for col in df.index.names if col != 'end'] , observed=True).cumsum()
    df = df.rename_axis(index={'end':'trade_date'})
    return df

def calc_optim_attrib_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_style).reset_index('style')
    if isinstance(df , pd.Series): df = df.to_frame()
    df = df.pivot_table('contribution' , df.index.names , columns='style' , observed=True).\
        rename_axis(None , axis='columns').sort_index().\
        groupby([col for col in df.index.names if col != 'end'] , observed=True).cumsum()
    df = df.rename_axis(index={'end':'trade_date'})
    return df
