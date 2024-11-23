import pandas as pd
from daily_bar.api import load_daily_bar_data


def calc_smpl_weight(root_path, target_weight_df, ecd):
    assert pd.Index(["TradeDate", "Code", "target_weight"]).difference(target_weight_df.columns).empty
    assert target_weight_df["TradeDate"].is_monotonic_increasing
    weight_sd = target_weight_df["TradeDate"].iloc[0]
    weight_data = target_weight_df.set_index(["TradeDate", "Code"])["target_weight"].unstack()
    #
    dbar_data = load_daily_bar_data(root_path, "basic", weight_sd, ecd)
    dbar_data = dbar_data[dbar_data['Code'].isin(weight_data.columns)].copy()
    dbar_ret = dbar_data.set_index(keys=['CalcDate', 'Code'])['ret'].unstack()
    #
    weight_data = weight_data.loc[:ecd, dbar_ret.columns].copy()
    assert weight_data.index.isin(dbar_ret.index).all()
    weight_adj_factor = (dbar_ret + 1.0).cumprod()
    weight_reb_factor = weight_adj_factor.loc[weight_data.index, :].reindex(weight_adj_factor.index).fillna(method='ffill')
    weight_adj_factor = weight_reb_factor / weight_adj_factor
    weight_data = weight_data.reindex(weight_adj_factor.index).fillna(method='ffill')
    weight_data = weight_data * weight_adj_factor
    weight_data = weight_data.div(weight_data.sum(axis=1), axis=0)
    rtn = weight_data[weight_data.index.to_series().between(weight_sd, ecd)].stack().rename('weight').reset_index(drop=False)
    rtn = rtn[rtn['weight'] > 0.0].copy()
    return rtn
