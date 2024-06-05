import os
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL


def parse_dict(data):
    rtn = dict()
    for key, val in data.items():
        assert "-" not in key
        if isinstance(val, dict):
            val = parse_dict(val)
            for new_key in val:
                rtn[key + "-" + new_key] = val[new_key]
        else:
            rtn[key] = val
    return rtn


def _save_job_results(root_save_path, job_rslts, save_job_list, stats_attributes, param_code):
    fcst_name = stats_attributes["forecast_name"]
    if "-" in fcst_name:
        fcst_shrt_name, preprocess_method = fcst_name.split("-")
    else:
        fcst_shrt_name, preprocess_method = fcst_name, "raw"
    shrt_bm_index = stats_attributes["bm_index"].split(":")[1].split("@")[0]
    result_save_path = os.path.join(root_save_path, stats_attributes["stock_universe"])
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    xlsx_name = "{0}@{1}@{2}@{3}@{4}.xlsx".format(fcst_name.replace('-', '_'), stats_attributes["port_start_date"],
                                                  stats_attributes["ret_end_date"], shrt_bm_index.split(".")[0], str(param_code))
    des_file_path = os.path.join(result_save_path, xlsx_name)
    stats_rslt_dict = OrderedDict()
    for job_name in save_job_list:
        if job_name == "stats_attributes":
            stats_rslt = pd.Series(parse_dict(stats_attributes))
            stats_rslt["bm_index"] = shrt_bm_index
            stats_rslt["forecast_name"] = stats_rslt["forecast_name"].replace("-", "_")
            stats_rslt.index.rename("param_name", inplace=True)
            stats_rslt = stats_rslt.to_frame(name="param_value").reset_index(drop=False)
            stats_rslt[["level_1", "level_2"]] = stats_rslt["param_name"].str.split("-", n=1, expand=True)
            stats_rslt["level_2"] = stats_rslt["level_2"].fillna(stats_rslt["level_1"])
            stats_rslt = stats_rslt[["level_1", "level_2", "param_value"]].copy()
            stats_rslt_dict[job_name] = stats_rslt
        elif job_name == "target_weight":
            stats_rslt = job_rslts["target_weight"]
            stats_rslt_dict[job_name] = stats_rslt
        elif job_name == "year_perf_rslts":
            stats_rslt = job_rslts["ret_analysis_job"][0]
            stats_rslt["forecast_short_name"] = fcst_shrt_name
            stats_rslt["preprocess_method"] = preprocess_method
            stats_rslt["bm_index"] = shrt_bm_index
            stats_rslt["trading_fee"] = stats_attributes["accnt_cfg"]["trading_fee"]
            stats_rslt["freq_type"] = stats_attributes["freq_type"]
            stats_rslt_dict[job_name] = stats_rslt
        elif job_name == "lag_perf_rslts":
            stats_rslt = job_rslts["lag_analysis_job"].set_index(["CalcDate", "lag_num"]).unstack()
            stats_rslt = pd.concat((stats_rslt[[("port_ret", "lag0"), ("index_ret", "lag0")]],
                                    stats_rslt[["ex_ret"]]), axis=1)
            stats_rslt.columns = stats_rslt.columns.to_flat_index()
            stats_rslt.rename(columns={col: "{0}_{1}".format(col[0], col[1]) for col in stats_rslt.columns},
                              errors="raise", inplace=True)
            stats_rslt.reset_index(drop=False, inplace=True)
            stats_rslt["forecast_short_name"] = fcst_shrt_name
            stats_rslt["preprocess_method"] = preprocess_method
            stats_rslt["bm_index"] = shrt_bm_index
            stats_rslt["freq_type"] = stats_attributes["freq_type"]
            stats_rslt_dict[job_name] = stats_rslt
        elif job_name == "win_rt":
            stats_rslt = job_rslts["ret_analysis_job"][2]
            stats_rslt_dict["组合胜率"] = stats_rslt
        elif job_name == "cum_perf":
            stats_rslt = job_rslts["ret_analysis_job"][1]
            stats_rslt = stats_rslt[['port_ret', 'index_ret', 'ex_ret']].copy()
            cum_perf = (stats_rslt[['port_ret', 'index_ret']] + 1).cumprod().rename(
                columns={"port_ret": "组合净值", "index_ret": "指数净值"}, errors="raise")
            last_dt = CALENDAR_UTIL.get_last_trading_dates([stats_rslt.index[0]], inc_self_if_is_trdday=False)[0]
            cum_perf.loc[last_dt] = 1.0
            cum_perf.sort_index(inplace=True)
            cum_perf["相对净值"] = cum_perf["组合净值"] / cum_perf["指数净值"]
            cum_perf.index = pd.to_datetime(cum_perf.index).date
            stats_rslt_dict["组合净值"] = cum_perf.reset_index(drop=False)
        elif job_name == "up_down_alpha":
            stats_rslt = job_rslts["ret_analysis_job"][3]
            stats_rslt_dict["上下行alpha"] = stats_rslt.reset_index(drop=False)
        elif job_name == "dcmp_ret":
            result = job_rslts["ret_attribution_job"][0]
            stats_rslt_dict["dcmp_ret"] = result.reset_index(drop=False)
        elif job_name == "style_attribution":
            result = job_rslts["ret_attribution_job"][1]
            stats_rslt_dict["style_attribution"] = result.reset_index(drop=False)
        elif job_name == "style_bias":
            result = job_rslts["risk_analysis_job"][0]
            stats_rslt_dict["style_bias"] = result.reset_index(drop=False)
        elif job_name == "industry_bias":
            result = job_rslts["risk_analysis_job"][1]
            stats_rslt_dict["industry_bias"] = result.reset_index(drop=False)
        elif job_name == "tracking_error":
            result = job_rslts["risk_analysis_job"][2]
            stats_rslt_dict["tracking_error"] = result.reset_index(drop=False)
        else:
            assert False, "  error::>>fmp_perf_tests>>save_job:{0} is unknown!".format(job_name)
    #
    with pd.ExcelWriter(des_file_path) as writer:
        for key, value in stats_rslt_dict.items():
            value.to_excel(writer, sheet_name=key, index=False)
    return des_file_path


def _gen_param_code(root_save_path, stats_attributes):
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    file_path = os.path.sep.join([root_save_path, "parameter_dict.xlsx"])
    new_param_info = pd.Series(parse_dict(stats_attributes)).drop(["forecast_name", "port_start_date",
                                                                   "ret_end_date", "stock_universe"])
    new_param_info = new_param_info.apply(lambda x: str(x) if isinstance(x, list) else x)
    new_param_info = new_param_info.to_frame().T
    if not os.path.exists(file_path):
        param_info = new_param_info.copy()
        param_info["param_code"] = 0
        param_code = 0
    else:
        local_param_info = pd.read_excel(file_path, sheet_name="参数字典")
        df = pd.concat((local_param_info, new_param_info), axis=0)
        if df.duplicated(subset=new_param_info.columns).any():
            param_info = local_param_info
            param_code = int(df.loc[df.duplicated(subset=new_param_info.columns, keep=False), "param_code"].dropna().values[0])
        else:
            param_code = max(local_param_info["param_code"]) + 1
            new_param_info["param_code"] = param_code
            param_info = pd.concat((local_param_info, new_param_info), axis=0)
    param_info = param_info[pd.Index(["param_code"]).append(param_info.columns.drop(["param_code"]))]
    param_info.to_excel(file_path, sheet_name="参数字典", index=False)
    return param_code


def _save_figures_in_pdf(root_save_path, stats_attributes, figures, param_code):
    fcst_name = stats_attributes["forecast_name"]
    shrt_bm_index = stats_attributes["bm_index"].split(":")[1].split("@")[0]
    pdf_name = "{0}@{1}@{2}@{3}@{4}.pdf".format(fcst_name.replace('-', '_'), stats_attributes["port_start_date"],
                                                stats_attributes["ret_end_date"],
                                                shrt_bm_index.split(".")[0], param_code)
    pdf_save_path = os.path.sep.join([root_save_path, stats_attributes["stock_universe"]])
    if not os.path.exists(pdf_save_path):
        os.makedirs(pdf_save_path)
    file_path = os.path.join(pdf_save_path, pdf_name)
    with PdfPages(file_path) as pdf:
        for fig in figures:
            pdf.savefig(fig[1])
            plt.close(fig[1])


def save_jobs(rslt, figures, save_config, stats_attributes):
    if save_config["suffix_param"] == "None":
        param_code = _gen_param_code(save_config["result_save_path"], stats_attributes)
        param_code = str(param_code)
    else:
        param_code = save_config["suffix_param"]
    _save_figures_in_pdf(save_config["result_save_path"], stats_attributes, figures, param_code)
    excel_rslt_file = _save_job_results(save_config["result_save_path"], rslt,  save_config["save_job_list"],
                                        stats_attributes, param_code)
    return excel_rslt_file