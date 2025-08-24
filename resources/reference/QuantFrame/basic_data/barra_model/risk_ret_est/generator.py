import os
import pandas as pd
from barra_model.factor_impl.api import load_barra_data
import numpy as np
from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
from .configs import get_risk_ret_path, get_special_ret_path
from crosec_mem.data_vendor import save


def prepare_ret_data(root_path, y_sd, y_ed):
    daily_bar = load_daily_bar_data(root_path, "basic", y_sd, y_ed)
    daily_bar = daily_bar[["CalcDate", "Code", "ret"]].copy()
    #
    date_df = daily_bar[["CalcDate"]].drop_duplicates()
    date_df["y_start"] = CALENDAR_UTIL.get_last_trading_dates(date_df["CalcDate"].tolist(), inc_self_if_is_trdday=False)
    date_df["CalcDate_x"] = CALENDAR_UTIL.get_last_dates(date_df["y_start"].tolist())
    ret_data = pd.merge(daily_bar, date_df[["CalcDate", "CalcDate_x"]], on=["CalcDate"], how="left")
    ret_data.rename(columns={"ret": "y"}, inplace=True, errors="raise")
    return ret_data


def _regress_factor_daily(df):
    industry_cols = df.columns[df.columns.str.contains('INDUSTRY')]
    country_style_cols = pd.Index(['COUNTRY']).append(df.columns[df.columns.str.contains('STYLE')])
    #
    industry_cnt = df[industry_cols].sum(axis=0)
    industry_names_chosen = industry_cnt[industry_cnt > 0.5].index.tolist()
    #
    industry_array = df[industry_names_chosen].values
    style_country_array = df[country_style_cols].values
    #
    industry_stk_weight = df["industry_stk_weight"].values
    industry_weight = industry_stk_weight.dot(industry_array)
    industry_weight = industry_weight / industry_weight.sum()
    #
    wls_weight = df["wls_weight"].values
    W = np.diag(wls_weight / wls_weight.sum())
    X = np.c_[style_country_array, industry_array]
    y = df["y"].values
    #
    X_adj = X[:, :-1].copy()
    X_adj[:, -len(industry_weight) + 1:] = \
        X_adj[:, -len(industry_weight) + 1:] - (
                    (industry_weight[:-1] / industry_weight[-1]).reshape(-1, 1) * X[:, -1]).T
    #
    beta_adj = np.linalg.inv(X_adj.T.dot(W).dot(X_adj)).dot(X_adj.T).dot(W).dot(y)
    last_beta = - (beta_adj[-len(industry_weight) + 1:] * industry_weight[:-1]).sum() / industry_weight[-1]
    beta = np.array(list(beta_adj) + [last_beta])
    beta = pd.Series(beta, index=pd.MultiIndex.from_product((['factor_ret'], country_style_cols.tolist() + industry_names_chosen), names=['asset_type', 'asset_ticker']))
    special_ret = pd.Series(y - X.dot(beta), index=pd.MultiIndex.from_product((['special_ret'], df["Code"]), names=['asset_type', 'asset_ticker']))
    rtn = pd.concat((beta, special_ret), axis=0).to_frame("risk_ret")
    return rtn


def get_float_cap(root_path, xsd, xed):
    fv_sd = CALENDAR_UTIL.get_last_trading_dates([xsd], inc_self_if_is_trdday=True)[0]
    float_value = load_daily_bar_data(root_path, "valuation", fv_sd, xed)[["CalcDate", "Code", "float_value"]]
    date_range = CALENDAR_UTIL.get_ranged_dates(fv_sd, xed)
    float_value = float_value.set_index(["CalcDate", "Code"])["float_value"].unstack().reindex(date_range). \
        fillna(method="ffill").stack().rename("float_value").reset_index(drop=False)
    float_value = float_value[float_value["CalcDate"].between(xsd, xed, inclusive="both")].copy()
    float_value.rename(columns={"CalcDate": "CalcDate_x"}, errors="raise", inplace=True)
    return float_value


def calc_risk_ret(root_path, barra_type, scd, ecd, style_flds=None):
    xsd, xed = CALENDAR_UTIL.get_last_dates(
        CALENDAR_UTIL.get_last_trading_dates([scd, ecd], inc_self_if_is_trdday=False))
    barra_data = load_barra_data(root_path, barra_type, xsd, xed)
    industry_col = barra_data.columns[barra_data.columns.str.contains('INDUSTRY')].copy()
    if style_flds is None:
        style_flds = barra_data.columns[barra_data.columns.str.contains('STYLE')].tolist()
    barra_data = barra_data[pd.Index(["CalcDate", "Code"]).append(industry_col).append(pd.Index(style_flds))].copy()
    barra_data = pd.get_dummies(barra_data, prefix='INDUSTRY', prefix_sep='.', columns=industry_col)
    barra_data["COUNTRY"] = 1
    #
    ret_data = prepare_ret_data(root_path, scd, ecd)
    all_data = pd.merge(barra_data.rename(columns={"CalcDate": "CalcDate_x"}, errors="raise"),
                        ret_data, on=["CalcDate_x", "Code"], how="inner")
    del barra_data, ret_data

    float_value = get_float_cap(root_path, xsd, xed)
    all_data = pd.merge(all_data, float_value, how="inner", on=["CalcDate_x", "Code"])
    all_data = all_data[all_data['float_value'] >= 1e7].copy()
    all_data["industry_stk_weight"] = all_data["float_value"]
    all_data["wls_weight"] = np.sqrt(all_data["float_value"])
    #
    dcm_results = all_data.groupby(["CalcDate"]).apply(_regress_factor_daily)
    dcm_results = dcm_results.swaplevel(i=0, j=1)
    risk_ret = dcm_results.loc['factor_ret']['risk_ret'].unstack()
    special_ret = dcm_results.loc['special_ret'].reset_index().rename(columns={'asset_ticker': 'Code', 'risk_ret': 'special_ret'}, errors='raise')
    #
    return risk_ret, special_ret


def gen_risk_ret(root_path, barra_type, scd, ecd):
    START_CALC_DATE = "2006-01-01"
    if scd < START_CALC_DATE:
        scd = START_CALC_DATE
        print("  warning::risk_ret>>generator>>start calc date is before {0}. change it to {0}.".format(
            START_CALC_DATE))
    scd = CALENDAR_UTIL.get_next_trading_dates([scd], inc_self_if_is_trdday=True)[0]
    ecd = CALENDAR_UTIL.get_last_trading_dates([ecd], inc_self_if_is_trdday=True)[0]
    assert scd <= ecd
    risk_ret, special_ret = calc_risk_ret(root_path, barra_type, scd, ecd)
    risk_ret_path = get_risk_ret_path(root_path, barra_type)
    special_ret_path = get_special_ret_path(root_path, barra_type)
    _save_risk_ret(risk_ret_path, risk_ret)
    save(special_ret_path, special_ret)


def _save_risk_ret(path, df):
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, "risk_ret.csv")
    if os.path.exists(file):
        y_data = pd.read_csv(file, index_col=0)
        df = pd.concat((y_data, df), axis=0)
        df = df[~df.index.duplicated()].copy()
    df.sort_index(inplace=True)
    df.to_csv(file)