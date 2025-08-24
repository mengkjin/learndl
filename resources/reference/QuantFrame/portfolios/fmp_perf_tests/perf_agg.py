import os
import pandas as pd


def load_single_factor_rslt(factor_test_rslt_excel, sheet_name_list=None):
    rtn = pd.read_excel(factor_test_rslt_excel, sheet_name=sheet_name_list)
    return rtn


def _agg_base_info_rslt(base_info_data, all_sample_rslts, fmp_excel_name):
    rtn = base_info_data.set_index(["level_2"])["param_value"]
    col_list = ['forecast_name', 'port_start_date', 'ret_end_date', 'bm_index', 'stock_universe'] +\
               rtn.index[rtn.index.str.contains("style")].tolist() + \
               ['industry', 'bm_weight_bound', 'exc_bounds', 'abs_bounds',
                'tracking_error-error_type', 'tracking_error-bound', 'turnover', 'LAMBDA', 'RHO',
                'risk_model_nm', 'init_cash', 'limit_trade_ratio', 'stamp_tax', 'trading_fee', 'freq_type', 'price_type']
    rtn = rtn[[nm for nm in col_list if nm in rtn.index]].copy()
    rtn.rename(index={"port_start_date": "start_date", "ret_end_date": "end_date"}, inplace=True)
    rtn["file_name"] = fmp_excel_name
    #
    rtn = pd.concat((all_sample_rslts[['绝对收益', '基准收益', '超额收益', '最大回撤', '相对最大回撤', '最大回撤天数',
                                       '最大回撤起始日期', '最大回撤结束日期', '跟踪误差', '信息比率', '相对收益回撤比', '换手率']], rtn), axis=0)
    rtn = rtn.to_frame().T.set_index("forecast_name")
    return rtn


def agg_single_factor_rslt(factor_test_rslt_excel, fmp_excel_name):
    hist_factor_rslt = load_single_factor_rslt(factor_test_rslt_excel,
                                               sheet_name_list=["stats_attributes", "year_perf_rslts", "lag_perf_rslts"])
    all_sample_rslts = hist_factor_rslt["year_perf_rslts"].set_index("年份").loc["全样本"]
    #
    base_info_data = _agg_base_info_rslt(hist_factor_rslt["stats_attributes"], all_sample_rslts, fmp_excel_name)
    rtn = {"base_info": base_info_data}
    return rtn


def save_agg_job_rslts(fmp_test_rslt_excel, agg_result_save_path, file_name):
    if not os.path.exists(agg_result_save_path):
        os.makedirs(agg_result_save_path)
    fmp_excel_name = os.path.basename(fmp_test_rslt_excel)
    fmp_excel_name = os.path.splitext(fmp_excel_name)[0]
    new_agg_rslt = agg_single_factor_rslt(fmp_test_rslt_excel, fmp_excel_name)
    #
    file_path = os.path.join(agg_result_save_path, "{0}.xlsx".format(file_name))
    if os.path.exists(file_path):
        all_agg_rslts = dict()
        hist_agg_rslts = pd.read_excel(file_path, index_col=0, sheet_name=None)
        all_agg_rslts["base_info"] = pd.concat((hist_agg_rslts["base_info"], new_agg_rslt["base_info"]), axis=0)
    else:
        all_agg_rslts = new_agg_rslt.copy()
    #
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name in ["base_info"]:
            all_agg_rslts[sheet_name].to_excel(writer, sheet_name=sheet_name, index=True)