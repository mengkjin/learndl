import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from daily_bar.api import load_daily_bar_data
from basic_src_data.wind_tools.basic import get_listed_ashare_codes_from_winddf
from barra_model.factor_impl.cne6.cne6_utils import apply_qtile_side_by_side_shrink


def calc_weight(root_path, scd, ecd):
    dscd = CALENDAR_UTIL.get_latest_n_trading_dates(scd, 20)[0]
    data = load_daily_bar_data(root_path, 'valuation', dscd, ecd)[['CalcDate', 'Code', 'total_value']]
    listed_codes = get_listed_ashare_codes_from_winddf(data['CalcDate'].iloc[0], data['CalcDate'].iloc[-1])
    data = pd.merge(listed_codes.rename(columns={'Date': 'CalcDate'}), data, how='inner', on=['CalcDate', 'Code'])
    data = data.set_index(['CalcDate', 'Code']).unstack().rolling(5).mean().stack()
    data = data.assign(log_val=np.log(data['total_value'])).reset_index()
    data = data[data['CalcDate'].between(scd, ecd)].copy()
    data = pd.merge(data, listed_codes.rename(columns={'Date': 'CalcDate'}), how='inner', on=['CalcDate', 'Code'])
    data.dropna(how='any', inplace=True)
    rtn = apply_qtile_side_by_side_shrink(data[['CalcDate', 'Code', 'log_val']].set_index(['CalcDate', 'Code']))
    rtn['est_val'] = np.exp(rtn['log_val'])
    rtn = rtn['est_val'] / rtn.groupby('CalcDate')['est_val'].sum()
    rtn.rename('size_weight', inplace=True)
    return rtn