from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd
from .data_center import DATAVENDOR


def _shift_calc_dates(calc_date_list, ecd, ret_range_type, lag_num):
    assert lag_num >= 0
    if ret_range_type == "period":
        fwd_shift_days = 1
        y_start = CALENDAR_UTIL.get_next_trading_dates(calc_date_list, inc_self_if_is_trdday=False, n=fwd_shift_days)
        assert len(set(y_start)) == len(y_start)
        y_start = [d for d in y_start if d < ecd]
        assert y_start
        y_end = y_start[1:] + [ecd]
        rtn = pd.concat((pd.Series(calc_date_list[: len(y_start)]).rename("CalcDate"),
                         pd.Series(y_start).rename("y_start").shift(-lag_num),
                         pd.Series(y_end).rename("y_end").shift(-lag_num)), axis=1).\
            dropna(subset=["y_start", "y_end"], how="any")
        assert rtn["CalcDate"].lt(rtn["y_start"]).all()
    else:
        assert False
    return rtn


def prepare_ret_data_with_lag(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type, price_type, lag,
                              keep_y_sd_ed=False):
    yend_ed = CALENDAR_UTIL.get_last_trading_dates([yend_ed], inc_self_if_is_trdday=True)[0]
    factor_val_df = factor_val_df.sort_values(by=["CalcDate", "Code"])
    date_df = _shift_calc_dates(factor_val_df["CalcDate"].drop_duplicates().tolist(), yend_ed, ret_range_type, lag)
    factor_val_lag = pd.merge(factor_val_df, date_df, how="inner", on=["CalcDate"])
    lag_ret = DATAVENDOR.query_period_data(factor_val_lag[["Code", "y_start", "y_end"]], ret_type, bm_index_nm, price_type)
    factor_ret_df = pd.merge(factor_val_lag, lag_ret, how="left", on=["Code", "y_start", "y_end"]).dropna(subset=["y"], how="any")
    if not keep_y_sd_ed:
        factor_ret_df.drop(columns=["y_start", "y_end"], inplace=True)
    return factor_ret_df


def init_data_center(root_path):
    DATAVENDOR.init_environments(root_path)