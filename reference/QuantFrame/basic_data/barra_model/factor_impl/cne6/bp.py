import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from daily_bar.api import load_daily_bar_data
from .cne6_utils import apply_qtile_shrink, apply_zscore, apply_qtile_side_by_side_shrink


def query_data(root_path, start_date, end_date):
    data = load_daily_bar_data(root_path, "valuation", start_date, end_date)[[
        'CalcDate', 'Code', 'book_value', 'total_value']].copy()
    data.rename(columns={'CalcDate': 'Date'}, inplace=True)
    data['Bp'] = data['book_value'] / data['total_value']
    data = data[['Date', 'Code', 'Bp']].copy()
    return data


def calc_bp(root_path, scd, ecd):
    data = query_data(root_path, scd, ecd)
    data.dropna(how='any', inplace=True)
    data["Bp"].clip(lower=0.0000001, inplace=True)
    assert data['Date'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    data.rename(columns={"Date": "CalcDate"}, errors="raise", inplace=True)
    rtn = data.dropna(how='any').set_index(['CalcDate', 'Code'])['Bp'].copy()
    rtn = np.log(rtn).rename('Bp')
    rtn = apply_qtile_side_by_side_shrink(rtn)
    return rtn


def process_bp(factor, weight):
    assert factor.index.get_level_values('CalcDate').drop_duplicates().tolist() == weight.index.get_level_values('CalcDate').drop_duplicates().tolist()
    all_data = pd.merge(np.log(factor), weight, how='inner', on=['CalcDate', 'Code'], sort=True)
    all_data['Bp'] = apply_qtile_side_by_side_shrink(all_data['Bp'])
    grp = all_data.groupby('CalcDate')
    rtn = (all_data['Bp'] - grp.apply(lambda x: (x['Bp'] * x['size_weight']).sum() / x['size_weight'].sum())) / grp['Bp'].std()
    rtn.rename('Bp', inplace=True)
    return rtn
