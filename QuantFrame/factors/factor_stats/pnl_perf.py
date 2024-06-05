import numpy as np
import pandas as pd
from .fcst_stats_utils.pnl_perf import calc_pnl


def calc_pnl_perf(factor_val_df, ret_type, price_type, yend_ed, bm_index_nm, indicator_list, grp_num=10):
    ret_range_type = "period"
    lag = 0
    pnl_perf, direction = calc_pnl(factor_val_df, ret_type, yend_ed, bm_index_nm, indicator_list,
                                   ret_range_type, price_type, lag, grp_num=grp_num)
    return pnl_perf, direction