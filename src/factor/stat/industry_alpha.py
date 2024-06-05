import pandas as pd
from industry.api import load_industry_data
from .fcst_stats_utils.ic_perf import calc_ic


def calc_industry_ic(root_path, factor_val_df, ret_type, price_type, industry_type, ecd, bm_index_nm, ic_type):
    industry_data = load_industry_data(
        root_path, factor_val_df["CalcDate"].min(),
        factor_val_df["CalcDate"].max(), industry_type, as_sys_id=False).rename(columns={industry_type: "industry"}, errors="raise")
    factor_val_df = pd.merge(factor_val_df, industry_data, how="inner", on=["CalcDate", "Code"])
    #
    ret_range_type = "period"
    lag = 0
    industry_ic = factor_val_df.groupby(["industry"]).apply(
        lambda x: calc_ic(x.drop(columns=["industry"]), ret_type, ecd, bm_index_nm, ret_range_type,
                          price_type, ic_type, lag))
    industry_ic.dropna(how="all", axis=0, inplace=True)
    industry_ic = industry_ic.reset_index(drop=False).set_index(["CalcDate"])
    #
    ic_mean = industry_ic.groupby(["industry"]).mean()
    ic_ir = (ic_mean / industry_ic.groupby(["industry"]).std())
    ic_stats = pd.concat((ic_mean.stack().rename("ic_mean"),
                          ic_ir.stack().rename("ic_ir")
                          ), axis=1, sort=True).reset_index(drop=False)
    return ic_stats