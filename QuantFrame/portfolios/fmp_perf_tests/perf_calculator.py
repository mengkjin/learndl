from port_backtester.deli_backtest import calc_portfolio_eod_info
from port_generals.weight_utils import calc_port_eod_weight, get_relative_port_stk_wt
from port_stats.ret_ts_analysor import evaluate_port_perf
from port_stats.lag_ts_analysor import calc_port_perf_with_lag
from port_stats.ts.risk_dcmp import calc_decomposed_port_nav
from port_stats.port_analysis import analyse_port_risk
from stk_index_utils.api import load_index_level
from stk_index_utils.api import calc_index_ret_bias
from datetime import datetime
import pandas as pd


def verify_attributes(target_weight, ecd, backtest_config, calc_config, fcst_nm, freq_type):
    assert pd.Index(["CalcDate", "Code"]).difference(target_weight.columns).empty
    assert target_weight["CalcDate"].is_monotonic_increasing
    rtn = dict()
    rtn["forecast_name"] = fcst_nm
    rtn["freq_type"] = freq_type
    rtn["port_start_date"] = target_weight["CalcDate"].iloc[0]
    rtn["port_end_date"] = target_weight["CalcDate"].iloc[-1]
    rtn["ret_end_date"] = ecd
    rtn["test_date"] = datetime.now().strftime('%Y-%m-%d')
    #
    rtn.update(backtest_config)
    rtn.update(calc_config)
    return rtn


def calc_ret_attribution_job(root_path, port_weight, eod_info_data, daily_cash, bm_index, barra_type, etd):
    nav = eod_info_data.groupby(["CalcDate"]).apply(lambda x: x["eod_share"].dot(x["eod_price"])) + daily_cash["eod_cash"]
    port_true_ret = nav / nav.shift(1) - 1
    index_level_data = load_index_level(root_path, port_true_ret.index[0], port_true_ret.index[-1], bm_index).set_index(["CalcDate"])
    index_ret = index_level_data["close_level"] / index_level_data["preclose_level"] - 1.0
    port_ex_ret = port_true_ret - index_ret
    #
    port_weight = port_weight.loc[port_weight["weight"].abs() > 1e-5, ["WeightDate", "Code", "weight"]].copy()
    port_weight = port_weight[port_weight["Code"] != "CNY"].copy()
    #
    rel_port_weight = get_relative_port_stk_wt(root_path, port_weight, bm_index)
    rel_port_weight = rel_port_weight[['WeightDate', 'Code', 'rel_weight']].rename(columns={"rel_weight": "weight"}, errors="raise")
    dcmp_ret = calc_decomposed_port_nav(root_path, rel_port_weight, barra_type, etd)
    index_ret_bias = calc_index_ret_bias(root_path, dcmp_ret.index[0], dcmp_ret.index[-1], bm_index)
    dcmp_ret['total'] = dcmp_ret['total'] + index_ret_bias
    dcmp_ret['Special'] = dcmp_ret['Special'] + index_ret_bias
    #
    dcmp_ret["true_ret"] = port_ex_ret.loc[dcmp_ret.index].copy()
    dcmp_ret["TradingCost"] = dcmp_ret["true_ret"] - dcmp_ret['total']
    dcmp_ret['Total'] = dcmp_ret['true_ret']
    #
    industry_dcmp = dcmp_ret.filter(regex="INDUSTRY", axis=1).sum(axis=1)
    style_dcmp = dcmp_ret.filter(regex="STYLE", axis=1).sum(axis=1)
    perf_attribution = pd.concat((
        dcmp_ret["COUNTRY"].rename("Country"),
        dcmp_ret["TradingCost"].rename("TradingCost"),
        industry_dcmp.rename("Industry"),
        style_dcmp.rename("Style"),
        dcmp_ret["Special"].rename("Special"),
        dcmp_ret["Total"].rename("Total")), axis=1)
    #
    style_attribution = dcmp_ret.filter(regex="STYLE", axis=1)
    style_attribution.columns = style_attribution.columns.str.replace("STYLE.", "", regex=False)
    return perf_attribution, style_attribution


def _calc_port_win_rt(perf_data):
    perf_data = perf_data.reset_index(drop=False)
    perf_data["CalcDate"] = pd.to_datetime(perf_data["CalcDate"])
    perf_data["week"] = perf_data["CalcDate"].dt.year.astype(str) + "-" + perf_data["CalcDate"].dt.isocalendar().week.astype(str)
    perf_data["month"] = perf_data["CalcDate"].dt.year.astype(str) + "-" + perf_data["CalcDate"].dt.month.astype(str)
    month_ret = perf_data.groupby(["month"])[['port_ret', 'index_ret', 'ex_ret']].sum()
    week_ret = perf_data.groupby(["week"])[['port_ret', 'index_ret', 'ex_ret']].sum()
    #
    month_win_rt = (month_ret["ex_ret"] > 0).sum() / len(month_ret)
    week_win_rt = (week_ret["ex_ret"] > 0).sum() / len(week_ret)
    rtn = pd.Series([month_win_rt, week_win_rt], index=["月度", "周度"]).to_frame("胜率").reset_index(drop=False).rename(
        columns={"index": "频率"})
    return rtn


def _calc_up_down_alpha(port_ret_df):
    port_ret_df = port_ret_df.copy()
    port_ret_df["grp"] = (port_ret_df["index_ret"] >= 0.0) * 1
    port_ret_df["grp"] = port_ret_df["grp"].replace({0: "下行", 1: "上行"})
    port_ret_df["year"] = port_ret_df.index.str[:4]
    all_sample_rslt = port_ret_df.groupby(["grp"])["ex_ret"].apply(lambda x: x.mean() * 245)
    year_rslt = port_ret_df.groupby(["year", "grp"])["ex_ret"].apply(lambda x: x.mean() * 245).unstack()
    year_rslt.loc["全样本", :] = all_sample_rslt
    return year_rslt


def calc_jobs(root_path, target_weights, ecd, env_config, backtest_config, calc_config, fcst_nm, freq_type):
    assert isinstance(target_weights, dict) and len(target_weights) == 1
    target_weight = list(target_weights.values())[0]
    #
    cash_df = 1 - target_weight.groupby(["TradeDate", "CalcDate"])["target_weight"].sum()
    cash_df = cash_df.reset_index(drop=False).assign(Code="CNY")
    target_weight = pd.concat((target_weight, cash_df), axis=0).sort_values(["TradeDate", "Code"]).reset_index(drop=True)
    #
    port_daily_weight = calc_port_eod_weight(root_path, target_weight, ecd)
    eod_info_data, cash_account = calc_portfolio_eod_info(
        root_path, backtest_config["init_cash"], port_daily_weight,
        backtest_config["trd_mtch_cfg"], backtest_config["accnt_cfg"], backtest_config["trd_make_cfg"])
    #
    results = dict()
    results['stats_attributes'] = verify_attributes(target_weight, ecd, backtest_config,
                                                    calc_config, fcst_nm, freq_type)
    results['target_weight'] = target_weight
    if calc_config['ret_analysis_job'] == "on":
        perf_rslt, port_index_df = evaluate_port_perf(root_path, eod_info_data, cash_account, backtest_config["bm_index"])
        win_rt = _calc_port_win_rt(port_index_df)
        up_down_alpha = _calc_up_down_alpha(port_index_df)
        results['ret_analysis_job'] = (perf_rslt, port_index_df, win_rt, up_down_alpha)
    if calc_config['lag_analysis_job'] == "on":
        results['lag_analysis_job'] = calc_port_perf_with_lag(
            root_path, target_weight, eod_info_data, cash_account, backtest_config["bm_index"], backtest_config["init_cash"],
            backtest_config["trd_mtch_cfg"], backtest_config["accnt_cfg"],
            backtest_config["trd_make_cfg"], calc_config["lag_num"], ecd)
    if calc_config['ret_attribution_job'] == "on":
        results['ret_attribution_job'] = calc_ret_attribution_job(
            root_path, port_daily_weight, eod_info_data, cash_account, backtest_config["bm_index"], env_config["risk_model_nm"], ecd)
    if calc_config['risk_analysis_job'] == "on":
        style_bias, industry_bias, tracking_error, constituent_ratio = analyse_port_risk(root_path, target_weight[target_weight['Code'] != 'CNY'], backtest_config["bm_index"], env_config["risk_model_nm"], 250)
        results['risk_analysis_job'] = (style_bias, industry_bias,tracking_error,constituent_ratio)
    return results