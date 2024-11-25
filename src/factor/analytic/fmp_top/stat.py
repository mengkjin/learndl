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
    df = filter_account(account).loc[:,['end','pf','bm','excess']]
    df = df.sort_values([*df.index.names , 'end']).rename(columns={'end':'trade_date'})
    df[['bm','pf']] = np.log(df[['bm','pf']] + 1)
    df[['bm','pf','excess']] = df.groupby(df.index.names , observed=True)[['bm','pf','excess']].cumsum()
    df[['bm','pf']] = np.exp(df[['bm','pf']]) - 1
    df = df.set_index('trade_date' , append=True)
    return df

def calc_top_perf_excess(account : pd.DataFrame):
    return calc_top_perf_curve(account)

def calc_top_perf_drawdown(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','excess']]
    df = df.sort_values([*df.index.names , 'end']).rename(columns={'end':'trade_date'})
    df['excess'] = df.groupby(df.index.names , observed=True)[['excess']].cumsum()
    df['peak']   = df.groupby(df.index.names , observed=True)[['excess']].cummax()
    df['drawdown'] = df['excess'] - df['peak']
    df = df.set_index('trade_date' , append=True).loc[:,['excess' , 'drawdown']]
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

def fetch_exp_style(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].style.loc[:,['active']]

def fetch_exp_indus(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].industry.loc[:,['active']]

def fetch_attrib_source(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].source.loc[:,['contribution']].rename_axis('source')

def fetch_attrib_style(x : pd.Series) -> pd.DataFrame:
    return x.iloc[0].style.loc[:,['contribution']].rename_axis('style')

def calc_top_exp_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_style).reset_index('style')
    df = df.pivot_table('active' , df.index.names , columns='style' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_top_exp_indus(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_indus).reset_index('industry')
    df = df.pivot_table('active' , df.index.names , columns='industry' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_top_attrib_source(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_source).reset_index('end',drop=True)
    df = df.groupby(df.index.names , observed=True).sum()
    return df

def calc_top_attrib_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_style).reset_index('end',drop=True)
    df = df.groupby(df.index.names , observed=True).sum()
    return df
