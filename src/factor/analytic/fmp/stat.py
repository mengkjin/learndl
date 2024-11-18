import time
import pandas as pd
import numpy as np

from typing import Any , Literal

from ....basic.conf import CATEGORIES_BENCHMARKS
from ....data import DATAVENDOR

def eval_drawdown(v : pd.Series | np.ndarray | Any , how : Literal['exp' , 'lin'] = 'lin'):
    if isinstance(v , np.ndarray): v = pd.Series(v)
    if how == 'lin':
        cum = v.cumsum() + 1.
        cummax = cum.cummax()
        cummdd = cummax - cum
    else:
        cum = (v + 1.).cumprod()
        cummax = cum.cummax()
        cummdd = 1 - cum / cummax
    return cummdd

def eval_drawdown_st(v : pd.Series | np.ndarray | Any , how : Literal['exp' , 'lin'] = 'lin'):
    if isinstance(v , np.ndarray): v = pd.Series(v)
    if how == 'lin':
        cum = v.cumsum() + 1.
    else:
        cum = (v + 1.).cumprod()
    return cum.expanding().apply(lambda x: x.argmax(), raw=True).astype(int)

def eval_max_drawdown(v : pd.Series | np.ndarray | Any , how : Literal['exp' , 'lin'] = 'lin'):
    dd , st = eval_drawdown(v , how) , eval_drawdown_st(v , how)
    mdd = dd.max()
    idx_ed = int(dd.argmax())
    idx_st = int(st.iloc[idx_ed])

    return mdd , idx_st , idx_ed

def eval_fmp_stats(grp : pd.DataFrame , mdd_period = True , **kwargs):
    period_len = abs(DATAVENDOR.CALENDAR.cd_diff(grp['start'].min() , grp['end'].max()))
    period_n   = len(grp)

    with np.errstate(divide = 'ignore'):
        pf_ret = np.prod(grp['pf'] + 1) - 1.
        bm_ret = np.prod(grp['bm'] + 1) - 1.
        excess = (pf_ret - bm_ret)
        ex_ann = np.power(np.prod(1 + grp['excess']) , 365 / period_len) - 1
        # pf_mdd = eval_max_drawdown(grp['pf'] , 'exp')
        ex_mdd , ex_mdd_st , ex_mdd_ed = eval_max_drawdown(grp['excess'] , 'lin')
        te     = np.std(grp['excess']) * np.sqrt(365 * period_n / period_len)
        ex_ir  = ex_ann / te
        ex_calmar = ex_ann / ex_mdd
        turn   = np.sum(grp['turn'])
        rslt = pd.DataFrame({'pf':pf_ret , 'bm':bm_ret , 'excess' : excess , 'annualized' : ex_ann , 'mdd' : ex_mdd , 
                             'te' : te , 'ir' : ex_ir , 'calmar' : ex_calmar , 'turnover' : turn} , index = [0])
    if mdd_period:
        rslt['mdd_period'] = ['{}-{}'.format(grp['end'].iloc[ex_mdd_st] , grp['end'].iloc[ex_mdd_ed])]
    return rslt.assign(**kwargs)

def calc_fmp_perf_period(account : pd.DataFrame , period : Literal['year' , 'yearmonth' , 'month'] = 'year'):
    if period=='year': account[period] = account['end'].astype(str).str[:4]
    elif period == 'yearmonth':  account[period] = account['end'].astype(str).str[:6]
    else: account[period] = account['end'].astype(str).str[4:6]
    group_cols = ['factor_name' , 'benchmark']
    account = account[account['lag'] == 0].sort_values('end')
    prd_stat = account.groupby(group_cols + [period]).\
        apply(eval_fmp_stats , mdd_period= (period != 'month') , include_groups = False).\
        reset_index(group_cols + [period]).reset_index(drop=True)
    all_stat = account.groupby(group_cols).\
        apply(eval_fmp_stats , mdd_period= (period != 'month') , include_groups = False).\
        reset_index(group_cols).reset_index(drop=True).assign(**{period:'ALL'})
    return pd.concat([prd_stat , all_stat])

def calc_fmp_perf_year(account : pd.DataFrame):
    return calc_fmp_perf_period(account , 'year')

def calc_fmp_perf_month(account : pd.DataFrame):
    return calc_fmp_perf_period(account , 'month')

def calc_fmp_perf_lag(account : pd.DataFrame):
    df = account
    df = df.loc[:,['factor_name','benchmark','end','excess','lag']].\
        set_index(['factor_name','benchmark','lag','end']).\
        groupby(['factor_name','benchmark','lag']).cumsum().\
        pivot_table(values='excess',index=['factor_name','benchmark','end'],columns=['lag'])
    df.columns = [f'lag{col}' for col in df.columns]
    if len(df.columns) == 2:
        cols = np.sort(df.columns.values)
        df['lag_cost'] = df[cols[0]] - df[cols[1]]
    return df.reset_index().rename(columns={'end':'trade_date'})

def calc_fmp_perf_curve(account : pd.DataFrame):
    df = account[account['lag']==0]
    df = df.loc[:,['factor_name','benchmark','end','pf','bm','excess']].\
        set_index(['factor_name','benchmark','end'])
    df[['bm','pf']] = np.log(df[['bm','pf']] + 1)
    df = df.groupby(['factor_name','benchmark'])[['bm','pf','excess']].cumsum()
    df[['bm','pf']] = np.exp(df[['bm','pf']]) - 1
    return df.reset_index().rename(columns={'end':'trade_date'})

def calc_fmp_perf_drawdown(account : pd.DataFrame):
    df = account[account['lag']==0]
    df = df.loc[:,['factor_name','benchmark','end','excess']].set_index(['factor_name','benchmark','end'])
    df = df.groupby(['factor_name','benchmark'])[['excess']].cumsum()
    peak = df.groupby(['factor_name','benchmark'])[['excess']].cummax()
    df['drawdown'] = df['excess'] - peak['excess']
    return df.reset_index().rename(columns={'end':'trade_date'})

def calc_fmp_style_exp(account : pd.DataFrame):
    df = account[(account['lag']==0) & (account['model_date']>0)]
    index_cols = ['factor_name','benchmark','start']
    df = df.loc[:,index_cols + ['analytic']].\
        set_index(index_cols).groupby(index_cols)['analytic'].\
        apply(lambda x:x.iloc[0].style.loc[:,['active']])
    df = df.pivot_table('active' , index_cols , columns='style').rename_axis(None , axis='columns')
    return df.reset_index().rename(columns={'start':'trade_date'})

def calc_fmp_industry_exp(account : pd.DataFrame):
    df = account[(account['lag']==0) & (account['model_date']>0)]
    index_cols = ['factor_name','benchmark','start']
    df = df.loc[:,index_cols + ['analytic']].\
        set_index(index_cols).groupby(index_cols)['analytic'].\
        apply(lambda x:x.iloc[0].industry.loc[:,['active']])
    df = df.pivot_table('active' , index_cols , columns='industry').rename_axis(None , axis='columns')
    return df.reset_index().rename(columns={'start':'trade_date'})

def calc_fmp_attrib_source(account : pd.DataFrame):
    index_cols = ['factor_name','benchmark','end']

    df0 = account[(account['lag']==0) & (account['model_date']<=0)].\
        groupby(index_cols)[['pf','bm']].min()
    df0 = df0.assign(tot = 0.).drop(columns=['pf','bm'])
    df1 = account[(account['lag']==0) & (account['model_date']>0)]
    df1 = df1.loc[:,index_cols + ['attribution']].\
        set_index(index_cols).groupby(index_cols)['attribution'].\
        apply(lambda x:x.iloc[0].source.loc[:,['contribution']].rename_axis('source')).\
        pivot_table('contribution' , index_cols , columns='source').rename_axis(None , axis='columns').\
        loc[:,['tot' , 'excess' , 'market' , 'industry' , 'style' , 'specific' , 'cost']]
    
    df = pd.concat([df0 , df1]).fillna(0).groupby(['factor_name','benchmark']).cumsum()
    return df.reset_index().rename(columns={'end':'trade_date'})

def calc_fmp_attrib_style(account : pd.DataFrame):
    index_cols = ['factor_name','benchmark','end']

    df0 = account[(account['lag']==0) & (account['model_date']<=0)].\
        groupby(index_cols)[['pf','bm']].min()
    df0 = df0.assign(size = 0.).drop(columns=['pf','bm'])

    df1 = account[(account['lag']==0) & (account['model_date']>0)]
    df1 = df1.loc[:,index_cols + ['attribution']].loc[:,index_cols + ['attribution']].\
        set_index(index_cols).groupby(index_cols)['attribution'].\
        apply(lambda x:x.iloc[0].style.loc[:,['contribution']].rename_axis('source')).\
        pivot_table('contribution' , index_cols , columns='source').rename_axis(None , axis='columns')
    
    df = pd.concat([df0 , df1]).fillna(0).groupby(['factor_name','benchmark']).cumsum()
    return df.reset_index().rename(columns={'end':'trade_date'})

def calc_fmp_prefix(account : pd.DataFrame):
    group_cols = ['factor_name' , 'benchmark'] 
    grouped = account[account['lag']==0].drop(columns=['lag']).groupby(group_cols)
    basic = pd.concat([grouped['start'].min() , grouped['end'].max()] , axis=1)
    stats = grouped.apply(eval_fmp_stats , include_groups=False).reset_index(group_cols).\
        reset_index(drop=True).set_index(group_cols)
    df = basic.join(stats).reset_index()
    df['benchmark'] = pd.Categorical(df['benchmark'] , categories = np.intersect1d(CATEGORIES_BENCHMARKS , df['benchmark']) , ordered=True) 
    df = df.sort_values(group_cols)
    return df
