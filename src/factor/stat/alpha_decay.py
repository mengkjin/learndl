import pandas as pd
import numpy as np
from .stats_utils.ic_perf import calc_ic
from .stats_utils.grouping import calc_group_perf


def calc_decay_pnl(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type, price_type, ic_type, lag_num):
    decay_pnl_df = []
    for lag in range(lag_num + 1):
        decay_pnl = calc_ic(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type, price_type, ic_type, lag)
        decay_pnl = decay_pnl.stack().rename("ic").reset_index(drop=False)
        decay_pnl["lag_type"] = "lag{0}".format(lag)
        decay_pnl_df.append(decay_pnl)
    decay_pnl_df = pd.concat(decay_pnl_df, axis=0)
    #
    ic_mean = decay_pnl_df.groupby(["factor_name", "lag_type"])["ic"].mean().rename("ic_mean").reset_index(drop=False)
    return ic_mean


def calc_decay_grp_perf(factor_val_df, ret_type, freq_type, yend_ed, grp_bm_index, group_nums, ret_range_type,
                        price_type, lag_num):
    decay_grp_perf = []
    for lag in range(lag_num + 1):
        grp_perf = calc_group_perf(factor_val_df, ret_type, yend_ed, grp_bm_index, group_nums, ret_range_type,
                                   price_type, lag)
        grp_perf["lag_type"] = "lag{0}".format(lag)
        decay_grp_perf.append(grp_perf)
    decay_grp_perf = pd.concat(decay_grp_perf, axis=0)
    #
    freq_days_dict = {"day": 245, "week": 50, "month": 12}
    assert freq_type in freq_days_dict.keys()
    decay_grp_ret_mean = decay_grp_perf.groupby(["factor_name", "group", "lag_type"])["group_ret"].mean() * freq_days_dict[freq_type]
    deacy_grp_rert_std = decay_grp_perf.groupby(["factor_name", "group", "lag_type"])["group_ret"].std() * \
                       np.sqrt(freq_days_dict[freq_type])
    decay_grp_ret_ir = decay_grp_ret_mean / deacy_grp_rert_std
    rtn = pd.concat(
        (
            decay_grp_ret_mean.rename("decay_grp_ret"),
            decay_grp_ret_ir.rename("decay_grp_ir")
         ),
        axis=1, sort=True).rename_axis("stats_name", axis="columns").stack().rename("stats_value").reset_index(drop=False)
    return rtn
