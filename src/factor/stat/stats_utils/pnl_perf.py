import numpy as np
import pandas as pd
from .data_preparation import prepare_ret_data_with_lag
from .weight_func import get_weights


def calc_pnl(factor_val_df, ret_type, yend_ed, bm_index_nm, weight_type_list, ret_range_type, price_type, lag, given_direction=None,
             grp_num=10):
    factor_ret_data = prepare_ret_data_with_lag(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type,
                                                price_type, lag)
    #
    factor_ret_data = factor_ret_data.drop(columns=['Code'])
    if given_direction is None:
        direction = np.sign(factor_ret_data.drop('CalcDate', axis=1).corr().loc["y"].drop("y"))
    else:
        direction = given_direction.copy()
    assert direction.index.equals(factor_ret_data.columns.drop(["CalcDate", "y"]))
    pnl = []
    for wt in weight_type_list:
        factor_results = factor_ret_data.groupby(by=['CalcDate'], as_index=True).apply(
            _calc_fake_pnl, weight_type=wt, ret_fld="y", direction=direction, grp_num=grp_num)
        factor_results.columns = pd.MultiIndex.from_product(([wt], factor_results.columns),
                                                            names=["stats_name", "factor_name"])
        pnl.append(factor_results)
    pnl = pd.concat(pnl, axis=1, sort=True)
    return pnl, direction


def _calc_fake_pnl(x, weight_type, ret_fld, direction, grp_num):
    fct_vals = x[x.columns.drop(["CalcDate", ret_fld])]
    rets = x[[ret_fld]].to_numpy()
    weights = get_weights(fct_vals, weight_type, direction, grp_num)
    rtn = (weights * rets).sum()
    return rtn