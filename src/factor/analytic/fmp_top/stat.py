import pandas as pd
import numpy as np

from typing import Any , Literal

from src.data import DATAVENDOR
from ..fmp_optim.stat import eval_pf_stats , filter_account

def calc_top_frontface(account : pd.DataFrame):
    df = filter_account(account)
    grouped = df.groupby(df.index.names , observed=True)
    basic = pd.concat([grouped['start'].min() , grouped['end'].max()] , axis=1)
    stats = grouped.apply(eval_pf_stats , include_groups=False).reset_index([None],drop=True)
    df = basic.join(stats).sort_index()
    return df

def calc_top_perf_curve(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','pf','bm','excess']].set_index('end' , append=True)
    df[['bm','pf']] = np.log(df[['bm','pf']] + 1)
    df = df.groupby(df.index.names , observed=True).cumsum()
    df[['bm','pf']] = np.exp(df[['bm','pf']]) - 1
    df = df.rename_axis(index={'end':'trade_date'})
    return df

def calc_top_perf_excess(account : pd.DataFrame):
    return calc_top_perf_curve(account)

def calc_top_perf_drawdown(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','excess']].sort_index()
    df['excess'] = df.groupby(df.index.names , observed=True)[['excess']].cumsum()
    df['peak']   = df.groupby(df.index.names , observed=True)[['excess']].cummax()
    df['drawdown'] = df['excess'] - df['peak']
    df = df.set_index('end' , append=True).loc[:,['excess' , 'drawdown']].rename_axis(index={'end':'trade_date'})
    return df

def calc_top_perf_period(account : pd.DataFrame , period : Literal['year' , 'yearmonth' , 'month'] = 'year'):
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

def calc_top_perf_year(account : pd.DataFrame):
    '''Calculate performance stats for each year'''
    return calc_top_perf_period(account , 'year')

def calc_top_perf_month(account : pd.DataFrame):
    '''Calculate performance stats for each calendar month'''
    return calc_top_perf_period(account , 'month')

def calc_top_exp_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(lambda x:x.iloc[0].style.loc[:,['active']]).\
        pivot_table('active' , df.index.names , columns='style' , observed=True).rename_axis(None , axis='columns')
    return df.rename_axis(index={'start':'trade_date'})

def calc_top_exp_indus(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(lambda x:x.iloc[0].industry.loc[:,['active']]).\
        pivot_table('active' , df.index.names , columns='industry' , observed=True).rename_axis(None , axis='columns')
    return df.rename_axis(index={'start':'trade_date'})

def calc_top_attrib_source(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].\
        apply(lambda x:x.iloc[0].source.loc[:,['contribution']].rename_axis('source'))
    df = df.groupby(df.index.names , observed=True).sum()
    return df

def calc_top_attrib_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].\
        apply(lambda x:x.iloc[0].style.loc[:,['contribution']].rename_axis('style'))
    df = df.groupby(df.index.names , observed=True).sum()
    return df
