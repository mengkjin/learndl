import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from index_weight.api import load_index_weight_data
# from stk_index_utils.api import load_index_weight_data


def get_index_components_filter(root_path, start_date, end_date):
    if start_date < '2005-04-08':
        start_date = '2005-04-08'
    data_start_date = CALENDAR_UTIL.get_last_trading_dates([start_date], inc_self_if_is_trdday=True)[0]
    df = load_index_weight_data(root_path, "broad_based", ['000300.SH', '000905.SH', '000852.SH', '932000.CSI', '000906.SH', '931865.CSI'],
                                data_start_date, end_date)
    exists_dates = df['CalcDate'].drop_duplicates()
    query_dates = CALENDAR_UTIL.get_ranged_dates(start_date, end_date)
    missed_dates = sorted(list(set(query_dates).difference(set(exists_dates))))
    latest_dates = [exists_dates[exists_dates < d].max() for d in missed_dates]
    date_maps = pd.DataFrame([missed_dates, latest_dates], index=['MissedDate', 'CalcDate']).T
    assert date_maps['CalcDate'].notna().all()
    missed_df = pd.merge(date_maps, df, how='left', on=['CalcDate']).drop(columns=['CalcDate']).rename(
        columns={'MissedDate': 'CalcDate'})
    rtn = pd.concat((df, missed_df), axis=0).sort_values(['CalcDate', 'Code'])
    rtn = rtn[rtn['CalcDate'].between(start_date, end_date)].reset_index(drop=True)
    rtn.columns = rtn.columns.str.replace('publish:', '')
    return rtn