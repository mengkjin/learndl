import pandas as pd
from .perf_calculator import calc_jobs
from .perf_showing import plot_jobs
from .perf_saver import save_jobs
from .perf_agg import save_agg_job_rslts
from port_builder.seq_portcons import calc_target_weights


def _get_configs(configs):
    env_config = configs["ENV_CONFIG"]
    backtest_config = configs["BACKTEST_CONFIG"]
    calc_config = configs["CALC_CONFIG"]
    show_config = configs["SHOW_CONFIG"]
    save_config = configs["SAVE_CONFIG"]
    if "OPT_CONFIG" in configs.keys():
        opt_config = configs["OPT_CONFIG"]
    else:
        opt_config = None
    return env_config, opt_config, backtest_config, calc_config, show_config, save_config


def _calc_prefix_data(opt_config, env_config):
    risk_param = opt_config["RISK"]
    rtn = [
        ("stock_universe", env_config["stock_universe"], "stock_universe"),
        ("risk_model_nm", env_config["risk_model_nm"], "risk_model_nm"),
        ("industry", risk_param["industry"], "risk"),
        ("bm_weight_bound", risk_param["bm_weight_bound"], "risk"),
        ("turnover_limit", risk_param["turnover"], "risk"),
        ("leverage", risk_param["leverage"], "risk"),
        ("{0}_te_limit".format(risk_param["tracking_error"]["error_type"]), risk_param["tracking_error"]["bound"], "risk"),
        ("stock_weight_abs_limit", risk_param["abs_bounds"], "risk"),
        ("stock_weight_relative_limit", risk_param["exc_bounds"], "risk"),
    ]
    rtn = pd.DataFrame(rtn, columns=["param_name", "Value", "Type"]).set_index("param_name")
    return rtn


def test_fmp_perf(root_path, fcst_df, ecd, configs, freq_type, initial_weight=None):
    assert pd.Index(["CalcDate", "Code"]).difference(fcst_df.columns).empty \
           and fcst_df.shape[1] == 3 and fcst_df["CalcDate"].is_monotonic_increasing
    fcst_nm = fcst_df.columns.drop(["CalcDate", "Code"])[0]
    fcst_df = fcst_df.rename(columns={fcst_nm: "fcst"}, errors="raise")
    env_config, opt_config, backtest_config, calc_config, show_config, save_config = _get_configs(configs)
    #
    target_weight = calc_target_weights(root_path, fcst_df, {"freq": "adapt_to_fcst"}, env_config, {"portfolio": opt_config},
                                        fcst_df["CalcDate"].iloc[0], fcst_df["CalcDate"].iloc[-1], initial_weight)
    #
    prefix_data = _calc_prefix_data(opt_config, env_config)
    stats_rslts = calc_jobs(root_path, target_weight, ecd, env_config, backtest_config, calc_config, fcst_nm,
                            freq_type)
    stats_attributes = stats_rslts["stats_attributes"]
    stats_attributes.update(env_config)
    stats_attributes.update(opt_config)
    #
    figures = plot_jobs(stats_rslts, show_config, prefix_data)
    #
    test_rslt_excel = save_jobs(stats_rslts, figures, save_config, stats_attributes)
    return test_rslt_excel

def test_fmp_perf_extended(root_path, fcst_df, target_weight, ecd, configs, freq_type, initial_weight=None):
    assert pd.Index(["CalcDate", "Code"]).difference(fcst_df.columns).empty \
           and fcst_df.shape[1] == 3 and fcst_df["CalcDate"].is_monotonic_increasing
    fcst_nm = fcst_df.columns.drop(["CalcDate", "Code"])[0]
    fcst_df = fcst_df.rename(columns={fcst_nm: "fcst"}, errors="raise")
    env_config, opt_config, backtest_config, calc_config, show_config, save_config = _get_configs(configs)
    #
    #target_weight = calc_target_weights(root_path, fcst_df, {"freq": "adapt_to_fcst"}, env_config, {"portfolio": opt_config},
    #                                    fcst_df["CalcDate"].iloc[0], fcst_df["CalcDate"].iloc[-1], initial_weight)
    #
    prefix_data = _calc_prefix_data(opt_config, env_config)
    stats_rslts = calc_jobs(root_path, target_weight, ecd, env_config, backtest_config, calc_config, fcst_nm,
                            freq_type)
    stats_attributes = stats_rslts["stats_attributes"]
    stats_attributes.update(env_config)
    stats_attributes.update(opt_config)
    #
    figures = plot_jobs(stats_rslts, show_config, prefix_data)
    #
    test_rslt_excel = save_jobs(stats_rslts, figures, save_config, stats_attributes)
    return test_rslt_excel

def analyse_port_perf(root_path, port_weight, ecd, configs, port_nm, freq_type):
    assert pd.Index(["CalcDate", "Code", "TradeDate", "target_weight"]).difference(port_weight.columns).empty \
           and port_weight["CalcDate"].is_monotonic_increasing
    env_config, opt_config, backtest_config, calc_config, show_config, save_config = _get_configs(configs)
    #
    stats_rslts = calc_jobs(root_path, {"portfolio": port_weight}, ecd, env_config, backtest_config, calc_config, port_nm,
                            freq_type)
    stats_attributes = stats_rslts["stats_attributes"]
    stats_attributes.update(env_config)
    #
    figures = plot_jobs(stats_rslts, show_config, prefix_data=None)
    #
    test_rslt_excel = save_jobs(stats_rslts, figures, save_config, stats_attributes)
    return test_rslt_excel


def agg_fmp_perf(factor_test_rslt_excel, agg_result_save_path, agg_file_name):
    save_agg_job_rslts(factor_test_rslt_excel, agg_result_save_path, agg_file_name)












