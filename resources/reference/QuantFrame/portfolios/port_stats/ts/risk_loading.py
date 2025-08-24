import pandas as pd
from barra_model.factor_impl.api import load_barra_data


def calc_port_risk_loading(root_path, port_info, barra_type):
    assert port_info.columns.equals(pd.Index(["CalcDate", "Code", "weight"]))
    calc_date_list = sorted(port_info["CalcDate"].unique())
    barra_data = load_barra_data(root_path, barra_type, calc_date_list[0], calc_date_list[-1])
    industry_col = barra_data.columns[barra_data.columns.str.contains('INDUSTRY')].copy()
    barra_data = pd.get_dummies(barra_data, prefix='INDUSTRY', prefix_sep='.', columns=industry_col)
    barra_data["COUNTRY"] = 1
    #
    stk_risk_loading = pd.merge(port_info, barra_data, how="left", on=["CalcDate", "Code"])
    if stk_risk_loading[barra_data.columns].isna().any(axis=1).any():
        print("  warning::port_stats>>ts>>risk_loading>>nan barra value found!")
    stk_risk_loading.set_index(["CalcDate", "Code"], inplace=True)
    factor_col = stk_risk_loading.columns.drop(["weight"])
    wt_col = "wt_" + factor_col
    stk_risk_loading[wt_col] = stk_risk_loading[factor_col].multiply(stk_risk_loading["weight"], axis=0)
    rtn = stk_risk_loading.groupby(by=["CalcDate"])[wt_col].sum().rename(
        columns=dict(zip(wt_col, factor_col)), errors="raise").reset_index(drop=False)
    return rtn


def calc_port_risk_loading_for_fund(root_path, port_info, barra_type):
    assert port_info.columns.equals(pd.Index(["CalcDate", "db_fund_key", "Code", "weight"]))
    calc_date_list = sorted(port_info["CalcDate"].unique())
    barra_data = load_barra_data(root_path, barra_type, calc_date_list[0], calc_date_list[-1])
    industry_col = barra_data.columns[barra_data.columns.str.contains('INDUSTRY')].copy()
    barra_data = pd.get_dummies(barra_data, prefix='INDUSTRY', prefix_sep='.', columns=industry_col)
    barra_data["COUNTRY"] = 1
    #
    stk_risk_loading = pd.merge(port_info, barra_data, how="left", on=["CalcDate", "Code"])
    if stk_risk_loading[barra_data.columns].isna().any(axis=1).any():
        print("  warning::port_stats>>ts>>risk_loading>>nan barra value found!")
    stk_risk_loading.set_index(["CalcDate", "db_fund_key", "Code"], inplace=True)
    factor_col = stk_risk_loading.columns.drop(["weight"])
    wt_col = "wt_" + factor_col
    stk_risk_loading[wt_col] = stk_risk_loading[factor_col].multiply(stk_risk_loading["weight"], axis=0)
    rtn = stk_risk_loading.groupby(by=["CalcDate", "db_fund_key"])[wt_col].sum().rename(
        columns=dict(zip(wt_col, factor_col)), errors="raise").reset_index(drop=False)
    return rtn