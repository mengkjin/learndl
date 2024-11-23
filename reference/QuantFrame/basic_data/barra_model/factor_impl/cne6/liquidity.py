from daily_bar.api import load_daily_bar_data
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from .cne6_utils import apply_zscore, apply_qtile_shrink


def calc_liquidity(root_path, scd, ecd):
    MONTH_TRADE_DAYS = 20
    QUARTER_TRADE_DAYS = 60
    YEAR_TRADE_DAYS = 245
    MIN_PRD_RATE = 0.5
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], n=YEAR_TRADE_DAYS, inc_self_if_is_trdday=False)[0]
    daily_bar = load_daily_bar_data(root_path, "basic", data_sd, ecd)
    daily_bar = daily_bar[(daily_bar["is_traded"] == 1) & (daily_bar['Code'].str[-2:] != 'BJ')].copy()
    turnover_data = daily_bar.set_index(["CalcDate", "Code"])["turnover"].unstack()
    #
    quarter_to = turnover_data.rolling(QUARTER_TRADE_DAYS, min_periods=int(QUARTER_TRADE_DAYS * MIN_PRD_RATE)
                                       ).mean() * MONTH_TRADE_DAYS
    year_to = turnover_data.rolling(YEAR_TRADE_DAYS, min_periods=int(YEAR_TRADE_DAYS * MIN_PRD_RATE)
                                    ).mean() * MONTH_TRADE_DAYS
    #
    liquidity = (quarter_to + year_to) / 2.0
    rtn = liquidity.loc[scd:, :].stack().rename("Liquidity").reset_index(drop=False)
    #
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code", "Liquidity"]].set_index(['CalcDate', 'Code'])
    assert (rtn['Liquidity'] > 0.0).all()
    rtn = apply_qtile_shrink(np.log(rtn))
    rtn = apply_zscore(rtn['Liquidity'].copy())
    return rtn
