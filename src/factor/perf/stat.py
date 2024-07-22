
import numpy as np
import pandas as pd

from typing import Any , Literal , Optional

from ..util import Benchmark , BENCHMARKS
from ...data import DataBlock
from ...data.vendor import DATAVENDOR 

def get_benchmark(benchmark : Optional[Benchmark | str] = None) -> Optional[Benchmark]:
    if isinstance(benchmark , str): benchmark = BENCHMARKS[benchmark]
    return benchmark

def factor_val_breakdown(factor_val : DataBlock | pd.DataFrame , 
                         benchmark : Optional[Benchmark | str] = None):
    benchmark = get_benchmark(benchmark)
    if benchmark: factor_val = benchmark(factor_val)
    if isinstance(factor_val , DataBlock):
        secid , date = factor_val.secid , factor_val.date
        factor_val = factor_val.to_dataframe()
    else:
        secid = factor_val.index.get_level_values('secid').unique().values
        date  = factor_val.index.get_level_values('date').unique().values
    return factor_val , secid , date

def get_fut_ret(factor_val : DataBlock | pd.DataFrame , nday : int = 10 , lag : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    factor_ret = DATAVENDOR.nday_fut_ret(secid , date , nday , lag , ret_type = ret_type).to_dataframe()
    factor_ret = factor_val.join(factor_ret , on = ['secid','date'])
    return factor_ret

def get_industry_exp(factor_val : DataBlock | pd.DataFrame):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    indus_block = DATAVENDOR.risk_industry_exp(secid , date)
    factor_ind = indus_block.to_dataframe().dropna(how = 'all')
    factor_ind = pd.DataFrame(factor_ind.idxmax(axis=1).rename('industry') , index = factor_ind.index)
    factor_ind = factor_val.join(factor_ind , on = ['secid','date'])
    return factor_ind

def eval_ic(factor_val : DataBlock | pd.DataFrame , nday : int = 10 , lag : int = 2 , 
            ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
            ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
    factor_val = get_fut_ret(factor_val , nday , lag , ret_type = ret_type)
    factor_list = factor_val.columns.values[:-1]
    ic = factor_val.groupby(by=['date'], as_index=True).apply(lambda x:x[factor_list].corrwith(x['ret'], method=ic_type))
    return ic.rename_axis('factor_name',axis='columns')

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

def eval_qtile_by_day(factor : pd.DataFrame , scaling : bool = True):
    if scaling: factor = (factor - factor.mean()) / factor.std()
    rtn = pd.concat([factor.quantile(q / 100).rename(f'{q}%') for q in (5,25,50,75,95)], axis=1, sort=True)
    return rtn.rename_axis('factor_name', axis='index')

def eval_stats(x):
    assert isinstance(x , pd.DataFrame) , type(x)
    x_sum = x.sum().rename('sum')
    x_avg = x.mean().rename('avg')
    x_std = x.std().rename('std')
    x_abs_avg = x.abs().mean().rename('abs_avg')
    x_ir = (x_avg / x_std).rename('ir')
    x_cumsum = x.cumsum()
    x_maxdown = (x_cumsum - x_cumsum.cummax()).min().rename('cum_mdd')
    return pd.concat([x_sum, x_avg , x_std, x_ir, x_abs_avg , x_maxdown], axis=1, sort=True)

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

def calc_decay_ic(factor_val : DataBlock | pd.DataFrame, nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                  benchmark : Optional[Benchmark | str] = None , ic_type : Literal['pearson' , 'spearman'] = 'pearson' , 
                  ret_type : Literal['close' , 'vwap'] = 'close'):
    '''
    nday : days of future return
    lag_init : starting lag of most recent future return , usually 1 or 2
    lag_num  : how many lagging period to calculate ic
    benchmark : within some benchmark
    '''
    decay_pnl_df = []
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    for lag in range(lag_num):
        decay_pnl = eval_ic(factor_val, nday , lag_init + lag * nday , ic_type = ic_type , ret_type = ret_type)
        decay_pnl_df.append(pd.DataFrame({'lag_type':f'lag{lag}','ic_mean':decay_pnl.mean()}))
    decay_pnl_df = pd.concat(decay_pnl_df, axis=0)
    return decay_pnl_df.reset_index()

def calc_grp_perf(factor_val : DataBlock | pd.DataFrame, benchmark : Optional[Benchmark | str] = None , 
                  nday : int = 10 , lag : int = 2 , group_num : int = 10 , excess = False , 
                  ret_type : Literal['close' , 'vwap'] = 'close' , trade_date = True) -> pd.DataFrame:
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    factor_val = get_fut_ret(factor_val , nday , lag , ret_type = ret_type)
    factor_list = factor_val.columns.values[:-1]
    df = factor_val.groupby('date').apply(eval_grp_avg , x_cols = factor_list , y_name = 'ret' , group_num = group_num , excess = excess) # type: ignore
    df = df.rename_axis('factor_name', axis='columns').stack().rename('group_ret').reset_index().sort_values(['date','group'])
    if trade_date:
        df['start'] = DATAVENDOR.td_offset(df['date'] , lag)
        df['end']   = DATAVENDOR.td_offset(df['date'] , lag + nday - 1)
    return df

def calc_decay_grp_perf(factor_val : DataBlock | pd.DataFrame, benchmark : Optional[Benchmark | str] = None , 
                        nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                        lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close'):
    decay_grp_perf = []
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    for lag in range(lag_num):
        grp_perf = calc_grp_perf(factor_val , nday = nday , lag = lag_init + lag * nday , 
                                 group_num = group_num , ret_type = ret_type , trade_date = False)
        decay_grp_perf.append(pd.DataFrame({'lag_type':f'lag{lag}',**grp_perf}))
    
    decay_grp_perf = pd.concat(decay_grp_perf, axis=0).reset_index()

    n_periods  = 243 / nday
    group_cols = ['factor_name', 'group', 'lag_type']
    decay_grp_ret_mean = decay_grp_perf.groupby(group_cols,observed=False)['group_ret'].mean() * n_periods
    deacy_grp_ret_std  = decay_grp_perf.groupby(group_cols,observed=False)['group_ret'].std() * np.sqrt(n_periods)
    decay_grp_ret_ir   = decay_grp_ret_mean / deacy_grp_ret_std
    rtn = pd.concat([decay_grp_ret_mean.rename('decay_grp_ret'),decay_grp_ret_ir.rename('decay_grp_ir')], axis=1, sort=True)
    rtn = pd.DataFrame(rtn.rename_axis('stats_name', axis='columns').stack() , columns=['stats_value'])
    return rtn.reset_index()

def calc_ic_monotony(factor_val : DataBlock | pd.DataFrame, benchmark : Optional[Benchmark | str] = None , 
                     nday : int = 10 , lag_init : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    grp_perf = calc_grp_perf(factor_val , nday = nday , lag = lag_init , group_num = 100 , 
                             excess=True , ret_type = ret_type , trade_date=False)

    n_periods  = 243 / nday
    group_cols = ['factor_name', 'group']
    grp_ret_mean = grp_perf.groupby(group_cols,observed=False)['group_ret'].mean() * n_periods
    grp_ret_std  = grp_perf.groupby(group_cols,observed=False)['group_ret'].std() * np.sqrt(n_periods)
    grp_ret_ir   = grp_ret_mean / grp_ret_std
    rtn = pd.concat([grp_ret_mean.rename('grp_ret'),grp_ret_ir.rename('grp_ir')], axis=1, sort=True)
    rtn = pd.DataFrame(rtn.rename_axis('stats_name', axis='columns').stack() , columns=['stats_value'])
    return rtn.reset_index()

def calc_style_corr(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    risk_style = DATAVENDOR.risk_style_exp(secid , date).to_dataframe()

    style_list , factor_list = risk_style.columns.tolist() , factor_val.columns.tolist()
    factor_style = pd.merge(factor_val, risk_style, on=['secid', 'date'], how='inner')

    factor_style_corr = factor_style.groupby('date' , observed=True).apply(
        lambda x: x.corr(method='spearman').loc[factor_list, style_list])
    factor_style_corr.index.rename(['date','factor_name'], inplace=True)

    return factor_style_corr.reset_index()

def calc_distribution(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None , 
                      sampling_date_num : int = 20 , hist_bins : int = 50):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    date_chosen = date[::int(np.ceil(len(date) / sampling_date_num))]
    factor_list = factor_val.columns.tolist()
    factor_val_date = factor_val.reset_index(drop = False).set_index('date')
    rtn = []
    for factor_name in factor_list:
        hist_dict = {}
        for date in date_chosen:
            factor_sample = factor_val_date.loc[date, [factor_name]].copy()
            cnts, bins = np.histogram(factor_sample, bins=hist_bins, density=False)
            hist_dict[date] = (cnts, bins)
        hist_df = pd.DataFrame(hist_dict, index=['hist_cnts', 'hist_bins']).T
        hist_df['factor_name'] = factor_name
        rtn.append(hist_df)
    rtn = pd.concat(rtn, axis=0)
    rtn.index.rename('date', inplace=True)
    rtn = rtn.reset_index(drop=False).loc[:,['date' , 'factor_name', 'hist_cnts', 'hist_bins']]
    return rtn

def calc_factor_qtile(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None , 
                      scaling : bool = True):
    factor_val , _ , _ = factor_val_breakdown(factor_val , benchmark)
    rtn = factor_val.groupby(['date']).apply(eval_qtile_by_day , scaling = scaling)
    return rtn.reset_index()

def calc_top_grp_perf_year(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None ,
                           nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                           ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    grp_perf = calc_grp_perf(factor_val , nday = nday , lag = lag , 
                             group_num = group_num , ret_type = ret_type , trade_date=False)
    top_group = grp_perf.groupby(['group', 'factor_name'] , observed=True)['group_ret'].sum().loc[
        [grp_perf['group'].min(), grp_perf['group'].max()]].reset_index(drop=False).sort_values(
        ['factor_name', 'group_ret']).drop_duplicates(['factor_name'], keep='last')

    top_perf = pd.merge(top_group[['factor_name', 'group']], grp_perf, how='left', on=['factor_name', 'group'])
    top_perf = top_perf.set_index(['date', 'factor_name'])['group_ret'].unstack()

    scd, ecd = str(top_perf.index[0]), str(top_perf.index[-1])
    factor_list = top_perf.columns.to_list()

    top_perf['year'] = top_perf.index.astype(str).str[:4]
    year_rslt = top_perf.groupby(['year'])[factor_list].apply(eval_stats).reset_index(drop=False)

    year_rslt['range'] = year_rslt['year'] + '0101-' + year_rslt['year'] + '1231'
    year_rslt.loc[year_rslt['year'] == scd[:4], 'range'] = f'{scd}-' + year_rslt['year'] + '1231'
    year_rslt.loc[year_rslt['year'] == ecd[:4], 'range'] = year_rslt['year'] + f'0101-{ecd}'

    full_rslt = eval_stats(top_perf[factor_list]).reset_index(drop=False)
    full_rslt['year'] = 'all'
    full_rslt['range'] = f'{scd}-{ecd}'

    n_periods  = 243 / nday

    rtn = pd.concat((year_rslt, full_rslt), axis=0)
    rtn['year_ret'] = rtn['avg'] * n_periods
    rtn['std'] = rtn['std'] * np.sqrt(n_periods)
    rtn['ir'] = rtn['ir'] * np.sqrt(n_periods)
    rtn = pd.merge(rtn, top_group[['factor_name', 'group']], on=['factor_name'], how='left')
    rtn = rtn.set_index(['factor_name', 'year', 'group' , 'range'])
    
    return rtn.reset_index()

def calc_ic_year(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None , 
                 nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                 ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    ic_df = eval_ic(factor_val , nday=nday , lag=lag , ic_type=ic_type , ret_type=ret_type)
    factor_list = ic_df.columns.tolist()

    ic_direction = ic_df.sum().apply(np.sign)
    ic_df = ic_df * ic_direction
    ic_df['year'] = ic_df.index.astype(str).str[:4]
    scd, ecd = str(ic_df.index[0]), str(ic_df.index[-1])

    year_rslt = ic_df.groupby(['year'])[factor_list].apply(eval_stats).reset_index(drop=False)
    year_rslt['range'] = year_rslt['year'] + '0101-' + year_rslt['year'] + '1231'
    year_rslt.loc[year_rslt['year'] == scd[:4], 'range'] = f'{scd}-' + year_rslt['year'] + '1231'
    year_rslt.loc[year_rslt['year'] == ecd[:4], 'range'] = year_rslt['year'] + f'0101-{ecd}'

    full_rslt = eval_stats(ic_df[factor_list]).reset_index(drop=False)
    full_rslt['year'] = 'all'
    full_rslt['range'] = f'{scd}-{ecd}'

    n_periods  = 243 / nday

    rtn = pd.concat((year_rslt, full_rslt), axis=0)
    rtn['year_ret'] = rtn['avg'] * n_periods
    rtn['std'] = rtn['std'] * np.sqrt(n_periods)
    rtn['ir'] = rtn['ir'] * np.sqrt(n_periods)

    rtn = pd.merge(rtn, ic_direction.rename('direction') , left_on=['factor_name'], right_index=True, how='left')
    rtn = rtn.set_index(['factor_name', 'year', 'direction' , 'range'])

    return rtn.reset_index()

def calc_ic_curve(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None , 
                  nday : int = 10 , lag : int = 2 ,  ma_windows : int | list[int] = [10,20] ,
                  ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                  ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    ic_df = eval_ic(factor_val , nday=nday , lag=lag , ic_type=ic_type , ret_type=ret_type)
    ic_df = ic_df.reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ic').set_index(['date','factor_name'])
    ic_curve = ic_df.join(ic_df.groupby('factor_name',observed=False).cumsum().rename(columns={'ic':'cum_ic'}))

    if isinstance(ma_windows , int): ma_windows = [ma_windows]
    grouped = ic_df.reset_index('factor_name').groupby('factor_name',observed=False)
    for ma in ma_windows:
        ic_curve = ic_curve.join(grouped.rolling(ma).mean().rename(columns={'ic':f'ma_{ma}'}))
    ic_curve = ic_curve
    return ic_curve.reset_index()

def calc_industry_ic(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None , 
                     nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                     ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    factor_ind = get_industry_exp(factor_val).reset_index().set_index(['secid','date','industry'])
    industry_ic = factor_ind.groupby(['industry']).apply(
            lambda x:eval_ic(x, nday=nday, lag=lag, ic_type=ic_type, ret_type=ret_type))
    ic_mean = industry_ic.groupby('industry').mean().stack()
    ic_std  = industry_ic.groupby('industry').std().stack()
    ic_ir   =  (ic_mean / ic_std)
    assert isinstance(ic_mean , pd.Series) and isinstance(ic_ir , pd.Series) , (ic_mean , ic_ir)
    ic_stats = pd.concat([ic_mean.rename('avg') , ic_ir.rename('ir')] , axis=1, sort=True)
    return ic_stats.reset_index()

def calc_pnl(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark | str] = None , 
             nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
             ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
             weight_type_list : list[str] = ['long' , 'long_short' , 'short']):
    factor_val , secid , date = factor_val_breakdown(factor_val , benchmark)
    factor_ret : pd.DataFrame = get_fut_ret(factor_val, nday , lag , ret_type = ret_type)

    if given_direction is None or given_direction == 0:
        direction = np.sign(factor_ret.corr().loc['ret'].drop('ret'))
    else:
        direction = given_direction

    pnl = []
    for wt in weight_type_list:
        factor_results = factor_ret.groupby('date').apply(lambda x:eval_weighted_pnl(x, wt, direction, group_num))
        factor_results = factor_results.reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ret')
        factor_results['weight_type'] = wt
        pnl.append(factor_results)

    pnl = pd.concat(pnl, axis=0).set_index(['weight_type' , 'date'])
    pnl = pnl.join(pnl.groupby(['factor_name' , 'weight_type']).cumsum().rename(columns={'ret':'cum_ret'})).reset_index()

    pnl_0 = pnl.groupby(['factor_name' , 'weight_type'])['date'].min().reset_index()
    pnl_0['cum_ret'] = 0.

    pnl['date'] = DATAVENDOR.td_offset(pnl['date'] , lag + nday - 1)
    pnl = pd.concat([pnl_0 , pnl])

    return pnl