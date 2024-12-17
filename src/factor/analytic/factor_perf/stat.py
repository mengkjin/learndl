
import numpy as np
import pandas as pd

from typing import Any , Literal , Optional

from src.data import DATAVENDOR 
from src.factor.util import Benchmark , StockFactor

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

def eval_ic_stats(ic_table : pd.DataFrame , nday : int = 5):
    n_periods  = 243 / nday
    
    ic_stats   = eval_stats(ic_table).reset_index(drop=False)

    ic_stats['direction'] = np.sign(ic_stats['avg'])
    ic_stats['year_ret'] = ic_stats['avg'] * n_periods
    ic_stats['std']      = ic_stats['std'] * np.sqrt(n_periods)
    ic_stats['ir']       = ic_stats['ir'] * np.sqrt(n_periods)

    melt_table = ic_table.reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='ic').dropna().reset_index()
    min_date = melt_table.groupby('factor_name')['date'].min()
    max_date = melt_table.groupby('factor_name')['date'].max()

    range_date = pd.concat([min_date.rename('start') , max_date.rename('end')] , axis=1).reset_index()
    range_date['range'] = range_date['start'].astype(str) + '-' + range_date['end'].astype(str)

    ic_stats = ic_stats.merge(range_date[['factor_name','range']] , on='factor_name')
    return ic_stats

def eval_qtile_by_day(factor : pd.DataFrame , scaling : bool = True):
    if scaling: factor = (factor - factor.mean()) / factor.std()
    rtn = pd.concat([factor.quantile(q / 100).rename(f'{q}%') for q in (5,25,50,75,95)], axis=1, sort=True)
    return rtn.rename_axis('factor_name', axis='index')

def calc_factor_frontface(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                          nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                          ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)
    ic_table = factor.eval_ic(nday , lag , ic_type , ret_type)
    ic_stats = eval_ic_stats(ic_table , nday = nday)
    return ic_stats

def calc_factor_coverage(factor : StockFactor , benchmark : Optional[Benchmark | str] = None):
    return factor.coverage(benchmark).reset_index().melt(id_vars=['date'],var_name='factor_name',value_name='coverage')

def calc_factor_ic_curve(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                         nday : int = 10 , lag : int = 2 ,  ma_windows : int | list[int] = [10,20] ,
                         ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                         ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)
    ic_table = factor.eval_ic(nday , lag , ic_type , ret_type).reset_index().\
        melt(id_vars=['date'],var_name='factor_name',value_name='ic').set_index(['date','factor_name'])
    ic_curve = ic_table.join(ic_table.groupby('factor_name',observed=False).cumsum().rename(columns={'ic':'cum_ic'}))

    if isinstance(ma_windows , int): ma_windows = [ma_windows]
    grouped = ic_table.reset_index('factor_name').groupby('factor_name',observed=False)
    for ma in ma_windows:
        ic_curve = ic_curve.join(grouped.rolling(ma).mean().rename(columns={'ic':f'ma_{ma}'}))
    ic_curve = ic_curve.reset_index()
    return ic_curve

def calc_factor_ic_decay(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                         nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                         ic_type : Literal['pearson' , 'spearman'] = 'pearson' , 
                         ret_type : Literal['close' , 'vwap'] = 'close'):
    '''
    nday : days of future return
    lag_init : starting lag of most recent future return , usually 1 or 2
    lag_num  : how many lagging period to calculate ic
    benchmark : within some benchmark
    '''
    factor = factor.within(benchmark)
    decay_pnl_df = []
    for lag in range(lag_num):
        decay_pnl = factor.eval_ic(nday , lag_init + lag * nday , ic_type , ret_type)
        decay_pnl_df.append(pd.DataFrame({'lag_type':f'lag{lag}','ic_mean':decay_pnl.mean()}))
    decay_pnl_df = pd.concat(decay_pnl_df, axis=0).reset_index()
    return decay_pnl_df

def calc_factor_ic_indus(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                         nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                         ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark) 
    ic_indus = factor.eval_ic_indus(nday , lag , ic_type , ret_type)
    ic_mean = ic_indus.groupby('industry').mean().stack()
    ic_std  = ic_indus.groupby('industry').std().stack()
    ic_ir   = ic_mean / (ic_std + 1e-6)
    assert isinstance(ic_mean , pd.Series) and isinstance(ic_ir , pd.Series) , (ic_mean , ic_ir)
    ic_stats = pd.concat([ic_mean.rename('avg') , ic_ir.rename('ir')] , axis=1, sort=True).reset_index()
    return ic_stats

def calc_factor_ic_year(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                        nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                        ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)
    ic_table = factor.eval_ic(nday , lag , ic_type , ret_type)
    
    full_rslt = eval_ic_stats(ic_table , nday = nday).assign(year = 'all')
    year_rslt = ic_table.assign(year = ic_table.index.astype(str).str[:4]).groupby(['year']).\
        apply(eval_ic_stats , nday = nday).reset_index(drop=False)

    rtn = pd.concat((year_rslt, full_rslt), axis=0)
    return rtn

def calc_factor_ic_benchmark(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                          nday : int = 10 , lag : int = 2 , ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
                          ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)
    ic_table = factor.eval_ic(nday , lag , ic_type , ret_type)
    ic_stats = eval_ic_stats(ic_table , nday = nday)
    return ic_stats

def calc_factor_ic_monotony(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                            nday : int = 10 , lag_init : int = 2 , ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)
    grp_perf = factor.eval_group_perf(nday , lag_init , 100 , True , ret_type , trade_date=False)
    
    grp_ret_mean = grp_perf.groupby(['factor_name', 'group'],observed=False)['group_ret'].mean() * 243 / nday
    grp_ret_std  = grp_perf.groupby(['factor_name', 'group'],observed=False)['group_ret'].std() * np.sqrt(243 / nday)
    grp_ret_ir   = grp_ret_mean / (grp_ret_std + 1e-6)
    rtn = pd.concat([grp_ret_mean.rename('grp_ret') , grp_ret_ir.rename('grp_ir')], axis=1, sort=True)
    rtn = pd.DataFrame(rtn.rename_axis('stats_name', axis='columns').stack() , columns=['stats_value']).reset_index()
    return rtn

def calc_factor_pnl_curve(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                          nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                          ret_type : Literal['close' , 'vwap'] = 'close' , given_direction : Literal[1,0,-1] = 0 ,
                          weight_type_list : list[str] = ['long' , 'long_short' , 'short']):
    factor = factor.within(benchmark)
    pnl = factor.eval_weighted_pnl(nday , lag , group_num , ret_type , given_direction , weight_type_list)
    pnl = pd.concat([pnl.groupby(['factor_name' , 'weight_type'])['date'].min().reset_index().assign(cum_ret = 0.) , pnl])
    pnl['date'] = DATAVENDOR.td_array(pnl['date'] , lag + nday - 1)
    return pnl

def calc_factor_style_corr(factor : StockFactor , benchmark : Optional[Benchmark | str] = None):
    factor = factor.within(benchmark)
    style = DATAVENDOR.risk_style_exp(factor.secid , factor.date).to_dataframe()
    df    = factor.frame().merge(style, on=['secid', 'date'], how='inner').groupby('date' , observed=True).\
        apply(lambda x: x.corr(method='spearman').loc[factor.factor_names, style.columns.values])
    df.index.rename(['date','factor_name'], inplace=True)
    df = df.reset_index()
    return df

def calc_factor_group_curve(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                            nday : int = 10 , lag : int = 2 , group_num : int = 10 , excess = False , 
                            ret_type : Literal['close' , 'vwap'] = 'close' , trade_date = True) -> pd.DataFrame:
    factor = factor.within(benchmark)
    grp_perf = factor.eval_group_perf(nday , lag , group_num , excess , ret_type , trade_date).set_index(['factor_name','group'])
    grp_perf0 = (grp_perf.groupby(grp_perf.index.names , observed=False)['date'].min() - 1).to_frame().assign(group_ret = 0.)
    df = pd.concat([grp_perf0 , grp_perf]).set_index('date' , append=True).sort_values(['group' , 'date']).\
        groupby(['factor_name','group'] , observed=True)['group_ret'].cumsum().rename('cum_ret').reset_index()
    return df

def calc_factor_group_decay(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                            nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                            lag_num : int = 5 , ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)

    decay_grp_perf = []
    for lag in range(lag_num):
        grp_perf = factor.eval_group_perf(nday , lag_init + lag * nday , group_num , True , ret_type , trade_date = False)
        decay_grp_perf.append(pd.DataFrame({'lag_type':f'lag{lag}',**grp_perf}))
    decay_grp_perf = pd.concat(decay_grp_perf, axis=0).reset_index()

    decay_grp_ret_mean = decay_grp_perf.groupby(['factor_name', 'group', 'lag_type'],observed=False)['group_ret'].mean() * 243 / nday
    deacy_grp_ret_std  = decay_grp_perf.groupby(['factor_name', 'group', 'lag_type'],observed=False)['group_ret'].std() * np.sqrt(243 / nday)
    decay_grp_ret_ir   = decay_grp_ret_mean / (deacy_grp_ret_std + 1e-6)
    rtn = pd.concat([decay_grp_ret_mean.rename('decay_grp_ret'),decay_grp_ret_ir.rename('decay_grp_ir')], axis=1, sort=True)
    rtn = pd.DataFrame(rtn.rename_axis('stats_name', axis='columns').stack() , columns=['stats_value'])
    return rtn.reset_index()

def calc_factor_group_year(factor : StockFactor , benchmark : Optional[Benchmark | str] = None ,
                           nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                           ret_type : Literal['close' , 'vwap'] = 'close'):
    factor = factor.within(benchmark)
    grp_perf = factor.eval_group_perf(nday , lag , group_num , True , ret_type , trade_date=False)
    top_group = grp_perf.groupby(['group', 'factor_name'] , observed=True)['group_ret'].sum().\
        loc[[grp_perf['group'].min(), grp_perf['group'].max()]].reset_index(drop=False).\
        sort_values(['factor_name', 'group_ret']).drop_duplicates(['factor_name'], keep='last')

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

    rtn = pd.concat((year_rslt, full_rslt), axis=0)
    rtn['year_ret'] = rtn['avg'] * 243 / nday
    rtn['std'] = rtn['std'] * np.sqrt(243 / nday)
    rtn['ir'] = rtn['ir'] * np.sqrt(243 / nday)
    rtn = pd.merge(rtn, top_group[['factor_name', 'group']], on=['factor_name'], how='left')
    rtn = rtn.set_index(['factor_name', 'year', 'group' , 'range']).sort_index().reset_index()
    return rtn

def calc_factor_distrib_curve(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , 
                              sampling_date_num : int = 20 , hist_bins : int = 50):
    factor = factor.within(benchmark)
    use_date = factor.date[::int(np.ceil(len(factor.date) / sampling_date_num))]
    rtn = []
    for factor_name in factor.factor_names:
        hist_dict = {}
        for date in use_date:
            factor_sample = factor.select(date = date , factor_name = factor_name).frame().iloc[:,0]
            cnts, bins = np.histogram(factor_sample, bins=hist_bins, density=False)
            hist_dict[date] = (cnts, bins)
        hist_df = pd.DataFrame(hist_dict, index=['hist_cnts', 'hist_bins']).T
        hist_df['factor_name'] = factor_name
        rtn.append(hist_df)
    rtn = pd.concat(rtn, axis=0)
    rtn.index.rename('date', inplace=True)
    rtn = rtn.reset_index(drop=False).loc[:,['date' , 'factor_name', 'hist_cnts', 'hist_bins']]
    return rtn

def calc_factor_distrib_qtile(factor : StockFactor , benchmark : Optional[Benchmark | str] = None , scaling : bool = True):
    factor = factor.within(benchmark)
    qtile = factor.frame().groupby(['date']).apply(eval_qtile_by_day , scaling = scaling).reset_index()
    return qtile