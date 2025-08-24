import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from daily_bar.api import load_daily_bar_data
from .cne6_utils import apply_qtile_side_by_side_shrink


def calc_linear_decay_ma(x_):
    cap = x_['total_value'].values
    cap = cap[np.logical_not(np.isnan(cap))]
    cap = cap[cap > 10000.0]
    if len(cap) > 0:
        w = np.arange(len(cap)) + 1
        rtn = np.sum(w * cap) / np.sum(w)
    else:
        rtn = np.nan
    return rtn


def query_data(root_path, start_date, end_date):
    data = load_daily_bar_data(root_path, "valuation", start_date, end_date)[['CalcDate', 'Code', 'total_value']].copy()
    data.dropna(how='any', inplace=True)
    data = data[data['total_value'] > 10000.0]
    data.rename(columns={'CalcDate': 'Date'}, inplace=True)
    data.set_index(keys=['Date'], inplace=True)
    return data


def calc_size(root_path, scd, ecd):
    data = query_data(root_path,
        str(int(scd[:4]) - 1) + scd[4:],
        ecd)
    #
    date_list = data.index.unique()
    rtn = list()
    for t in range(180, len(date_list)):
        this_d = date_list[t]
        if scd <= this_d <= ecd:
            period_data = data.loc[date_list[t - 20 + 1]: date_list[t], :]
            this_d_data = data.loc[date_list[t], :]
            rslt = period_data.groupby(["Code"], as_index=True).apply(calc_linear_decay_ma)
            rslt.dropna(inplace=True)
            rslt = pd.DataFrame(rslt, columns=['total_value']).reset_index(drop=False)
            rslt = pd.merge(this_d_data[['Code']], rslt, how='inner', on=['Code'])
            log_cap = np.log(rslt['total_value'].values)
            rslt['Size'] = log_cap
            rslt['Date'] = this_d
            rslt = rslt[['Date', 'Code', 'Size']]
            rtn.append(rslt)
    rtn = pd.concat(rtn, axis=0)
    rtn = rtn[rtn['Date'].between(scd, ecd)].copy()
    assert rtn['Date'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    rtn.rename(columns={"Date": "CalcDate"}, errors="raise", inplace=True)
    rtn = rtn.dropna(how='any').set_index(['CalcDate', 'Code'])['Size'].copy()
    return rtn


def process_size(factor, weight):
    assert factor.index.get_level_values('CalcDate').drop_duplicates().tolist() == weight.index.get_level_values('CalcDate').drop_duplicates().tolist()
    all_data = pd.merge(factor, weight, how='inner', on=['CalcDate', 'Code'], sort=True)
    all_data['Size_wrz'] = apply_qtile_side_by_side_shrink(all_data['Size'], m=2.0)
    grp = all_data.groupby('CalcDate')
    rtn = (all_data['Size_wrz'] - grp.apply(lambda x: (x['Size_wrz'] * x['size_weight']).sum() / x['size_weight'].sum())) / grp['Size_wrz'].std()
    rtn.rename('Size', inplace=True)
    rtn.clip(lower=-3.0, upper=2.0, inplace=True)
    rtn = rtn / rtn.groupby('CalcDate').std()
    return rtn


from sklearn.linear_model import LinearRegression


def f2(x):
    y = (x['Size'] / 3.0).to_numpy() ** 3
    lm = LinearRegression().fit(x['Size'].to_numpy().reshape((-1, 1)), y, sample_weight=np.sqrt(x['size_weight'].to_numpy()))
    res = y - lm.predict(x['Size'].to_numpy().reshape((-1, 1)))
    res_mean = np.sum(x['size_weight'] * res) / np.sum(x['size_weight'])
    return pd.Series((res - res_mean) / np.std(res), index=x.index)


def calc_nonlinear_size(size_factor, weight):
    all_data = pd.merge(size_factor, weight, how='left', on=['CalcDate', 'Code'])
    assert all_data['size_weight'].notnull().all()
    if all_data.index.get_level_values("CalcDate").nunique() == 1:
        rtn = f2(all_data)
    else:
        rtn = all_data.groupby('CalcDate', as_index=False, group_keys=False).apply(f2)
    rtn.rename('Nonlin_Size', inplace=True)
    return rtn