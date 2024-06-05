from daily_bar.api import load_daily_bar_data
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from .cne6_utils import apply_qtile_shrink, apply_zscore


def calc_mom(root_path, scd, ecd):
    ROLLING_WINSIZE = 245
    LAG_WINSIZE = 11
    LAG_DAYS = 11
    HALF_LIFE = 122
    MIN_PRD_RATE = 0.5
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], n=ROLLING_WINSIZE + LAG_DAYS + LAG_WINSIZE + 60, inc_self_if_is_trdday=False)[0]
    daily_bar = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    daily_bar = daily_bar[(daily_bar["is_traded"] == 1) & (daily_bar['Code'].str[-2:] != 'BJ')].copy()
    lg_ret_df = daily_bar.set_index(["CalcDate", "Code"])["log_ret"].unstack()
    #
    alpha = 1 - np.power(0.5, 1 / HALF_LIFE)
    ones_ewm = lg_ret_df.mask(lg_ret_df.notna(), other=1.0).ewm(halflife=HALF_LIFE, min_periods=1, adjust=False, ignore_na=False).mean()
    ewm_weight = (ones_ewm - ones_ewm.shift(ROLLING_WINSIZE) * np.power(1 - alpha, ROLLING_WINSIZE)) / alpha
    lg_ret_ewm = lg_ret_df.ewm(halflife=HALF_LIFE, min_periods=1, adjust=False, ignore_na=False).mean()
    rs_df = (lg_ret_ewm - lg_ret_ewm.shift(ROLLING_WINSIZE) * np.power(1 - alpha, ROLLING_WINSIZE)) / alpha
    rs_df = rs_df / ewm_weight
    ret_count = lg_ret_df.rolling(ROLLING_WINSIZE, min_periods=1).count()
    rs_df = rs_df.mask(ret_count <= MIN_PRD_RATE*ROLLING_WINSIZE, other=np.nan)
    #
    rstr = rs_df.rolling(LAG_WINSIZE, min_periods=int(MIN_PRD_RATE * LAG_WINSIZE)).mean().shift(LAG_DAYS)
    rtn = rstr.stack().rename("rstr").reset_index(drop=False)
    #
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both")].set_index(['CalcDate', 'Code'])
    rtn = apply_qtile_shrink(rtn)
    rtn = apply_zscore(rtn['rstr'].rename('Momentum'))
    return rtn