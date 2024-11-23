import pandas as pd
from .data_preparation import prepare_ret_data_with_lag


def _calc_grp_avg(x, x_col, y_nm, group_num):
    y = pd.DataFrame(x[y_nm], index=x.index, columns=[y_nm])
    rtn = list()
    for x_nm in x_col:
        y['group'] = pd.qcut(x[x_nm], group_num,
                             labels=["group{0}".format(i) for i in range(1, group_num + 1)])
        grp_avg_ret = y.groupby('group')[y_nm].mean().rename(x_nm)
        rtn.append(grp_avg_ret)
    rtn = pd.concat(rtn, axis=1, sort=True)
    return rtn


def calc_group_perf(factor_val_df, ret_type, yend_ed, grp_bm_index, group_nums, ret_range_type, price_type, lag):
    factor_ret_data = prepare_ret_data_with_lag(factor_val_df, ret_type, yend_ed, grp_bm_index, ret_range_type,
                                                price_type, lag)
    #
    x_col = factor_ret_data.columns.drop(['CalcDate', 'Code', 'y'])
    df = factor_ret_data.groupby('CalcDate').apply(_calc_grp_avg, x_col=x_col, y_nm="y", group_num=group_nums)
    rtn = df.rename_axis("factor_name", axis="columns").stack().rename("group_ret").reset_index().\
        sort_values(["CalcDate", "group"])
    return rtn