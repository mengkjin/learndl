import pandas as pd
from divnsplit.api import load_divnsplit_data
from events_system.calendar_util import CALENDAR_UTIL


def calc_volume_adj_factor(root_path, data):
    assert data["CalcDate"].is_monotonic_increasing
    scd, ecd = data["CalcDate"].iloc[0], data["CalcDate"].iloc[-1]
    divnsplit_data = load_divnsplit_data(root_path, scd, ecd)
    split_ratio = divnsplit_data.set_index(["ExDate", "Code"])["split_ratio"].unstack()
    split_ratio = split_ratio.reindex(pd.Index(CALENDAR_UTIL.get_ranged_dates(scd, ecd), name="CalcDate"))
    split_ratio.fillna(0.0, inplace=True)
    adj_factor = (1.0 / (split_ratio + 1.0)).cumprod()
    adj_factor = adj_factor.divide(adj_factor.iloc[-1], axis=1)
    adj_factor = adj_factor.stack().rename('adj_factor').reset_index(drop=False)
    rtn = pd.merge(data, adj_factor, how='left', on=['CalcDate', 'Code'])
    rtn['adj_factor'].fillna(1.0, inplace=True)
    return rtn