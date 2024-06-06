import numpy as np
import pandas as pd
from .data_preparation import prepare_ret_data_with_lag

def calc_ic(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type, price_type, ic_type, lag):
    assert ic_type in ("pearson", "spearman")
    factor_ret_data = prepare_ret_data_with_lag(factor_val_df, ret_type, yend_ed, bm_index_nm, ret_range_type,
                                                price_type, lag)
    #
    factor_list = factor_ret_data.columns.drop(["CalcDate", "Code", "y"]).tolist()
    rtn = factor_ret_data.groupby(by=['CalcDate'], as_index=True).apply(
        lambda x: x[factor_list].corrwith(x["y"], method=ic_type))
    rtn.columns.rename("factor_name", inplace=True)
    return rtn