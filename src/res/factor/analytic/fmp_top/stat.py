import pandas as pd

from typing import Any , Literal

from ...util.stat import eval_pf_stats , eval_cum_ret , eval_drawdown
from ...util.agency import BaseConditioner

def filter_account(account : pd.DataFrame , lag0 = True , pos_model_date = False):
    '''drop lag if exists , and select lag0'''
    if lag0:
        if 'lag' in account.index.names: 
            account = account.reset_index('lag',drop=False)
        if 'lag' in account.columns: 
            account = account.query('lag==0').drop(columns=['lag'])
    if pos_model_date:
        account = account.query('model_date>0')
    return account

def calc_top_frontface(account : pd.DataFrame):
    df = filter_account(account)
    grouped = df.groupby(df.index.names , observed=True)
    start : pd.Series | Any = grouped['start'].min()
    end : pd.Series | Any = grouped['end'].max()
    basic = pd.concat([start , end] , axis=1)
    stats = grouped.apply(eval_pf_stats , include_groups=False).reset_index([None],drop=True)
    df = basic.join(stats).sort_index()
    return df

def calc_top_perf_curve(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','pf','bm','excess']]
    index_names = [str(name) for name in df.index.names]
    df = df.sort_values([*index_names , 'end']).rename(columns={'end':'trade_date'})
    df[['bm','pf']] = eval_cum_ret(df[['bm','pf']] , 'exp' , groupby=index_names)
    df['excess'] = eval_cum_ret(df['excess'] , 'lin' , groupby=index_names)
    df = df.set_index('trade_date' , append=True)
    return df

def calc_top_perf_excess(account : pd.DataFrame):
    return calc_top_perf_curve(account)

def calc_top_perf_drawdown(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','pf','overnight']]
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
            sub[conditioner.conditioner_name()] = conditioner.conditioned_pf_ret(sub , plot = False)
        sub = sub.set_index('trade_date' , append=True).drop(columns=['pf','overnight'])
        dfs.append(sub)
    df = pd.concat(dfs)

    '''
    df = eval_detailed_drawdown(df['pf'] , groupby = index_names)

    df['warning'] = (df['recover_ratio'].fillna(0) < 0.25) * (df['uncovered_max_drawdown'] < -0.1) * 1.0
    df['stopped'] = df['pf'] * (1 - df.groupby(index_names , observed=True , group_keys = False)['warning'].shift(1).fillna(0.))
    df['stopped'] = eval_cum_ret(df['stopped'] , 'exp' , groupby = index_names)
    df['trade_date'] = trade_date
    df = df.set_index('trade_date' , append=True).drop(columns=['pf']).rename(columns={'cum_ret':'pf'})
    '''
    return df

def calc_top_perf_excess_drawdown(account : pd.DataFrame):
    df = filter_account(account).loc[:,['end','excess']]
    index_names = [str(name) for name in df.index.names]
    df = df.sort_values([*index_names , 'end']).rename(columns={'end':'trade_date'})
    df['excess'] = eval_cum_ret(df['excess'] , 'lin' , groupby=index_names)
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
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_style , include_groups = False).reset_index('style')
    df = df.pivot_table('active' , df.index.names , columns='style' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_top_exp_indus(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['start','analytic']].set_index('start' , append=True)
    df = df.groupby(df.index.names , observed=True)['analytic'].apply(fetch_exp_indus , include_groups = False).reset_index('industry')
    df = df.pivot_table('active' , df.index.names , columns='industry' , observed=True)
    df = df.rename_axis(None , axis='columns').rename_axis(index={'start':'trade_date'})
    return df

def calc_top_attrib_source(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_source , include_groups = False).reset_index('end',drop=True)
    df = df.groupby(df.index.names , observed=True).sum()
    return df

def calc_top_attrib_style(account : pd.DataFrame):
    df = filter_account(account , pos_model_date=True)
    df = df.loc[:,['end','attribution']].set_index('end' , append=True)
    df = df.groupby(df.index.names , observed=True)['attribution'].apply(fetch_attrib_style , include_groups = False).reset_index('end',drop=True)
    df = df.groupby(df.index.names , observed=True).sum()
    return df
