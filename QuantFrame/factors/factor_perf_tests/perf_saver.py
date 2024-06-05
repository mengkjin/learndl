import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _save_job_results(root_save_path, job_rslts, save_job_list, stats_attributes):
    factor_name = stats_attributes["factor_name"][0]
    factor_name_splits = factor_name.split('-')
    assert len(factor_name_splits) == 2
    factor_shrt_name, preprocess_method = factor_name_splits
    freq = stats_attributes['freq_type']
    result_save_path = os.path.sep.join([root_save_path, stats_attributes["stock_universe_type"], factor_name.split('-')[0]])
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    xlsx_name = "{0}@{1}@{2}.xlsx".format(factor_name.replace('-', '_'), stats_attributes["factor_start_date"],
                                          stats_attributes["ret_end_date"])
    des_file_path = os.path.join(result_save_path, xlsx_name)
    with pd.ExcelWriter(des_file_path) as writer:
        for job_cate, job_name in save_job_list:
            job_result = job_rslts[job_cate]
            if job_cate == "pnl_job":
                if job_name == "long_short_curve":
                    stats_rslt = job_result[job_name].stack().reset_index(drop=False)
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["pnl_bm_index"] = stats_attributes["pnl_bm_index"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                elif job_name == "ic_curve":
                    stats_rslt = job_result["ic_curve"][0].stack().rename("ic").reset_index(drop=False)
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["pnl_bm_index"] = stats_attributes["pnl_bm_index"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                elif job_name == "ic_results":
                    stats_rslt = job_result["ic_curve"][1].reset_index(drop=False)
                    stats_rslt["factor_name"] = stats_attributes["factor_name"][0]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                    stats_rslt["freq"] = freq
                    stats_rslt["pnl_bm_index"] = stats_attributes["pnl_bm_index"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                elif job_name == "ic_yearly_results":
                    stats_rslt = job_result["ic_curve"][2]
                    stats_rslt["direction"] = stats_rslt["direction"].astype(int).replace({1: "正向", -1: "负向"})
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                    stats_rslt["freq"] = freq
                    stats_rslt["pnl_bm_index"] = stats_attributes["pnl_bm_index"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                else:
                    assert False, "  error::>>factor_perf_tests>>pnl_stats:{0} is unknown!".format(job_name)
            elif job_cate == "group_job":
                if job_name == "group_perf_curve":
                    stats_rslt = job_result["group_perf_curve"][0]
                    stats_rslt["group"] = stats_rslt["group"].astype(str)
                    group_names = list(stats_rslt["group"].unique())
                    stats_rslt = stats_rslt.set_index(["CalcDate", "factor_name", "group"])["group_ret"].unstack().\
                        reset_index(drop=False)
                    stats_rslt = stats_rslt[["CalcDate", "factor_name"] + group_names].copy()
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["group_bm_index"] = stats_attributes["group_bm_index"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                elif job_name == "top_grp_perf_rslts":
                    stats_rslt = job_result["group_perf_curve"][1]
                    stats_rslt["group"] = stats_rslt["group"].astype(str)
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["group_bm_index"] = stats_attributes["group_bm_index"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                else:
                    assert False, "  error::>>factor_perf_tests>>group_stats:{0} is unknown!".format(job_name)
            elif job_cate == "density_job":
                if job_name == "factor_distribution":
                    stats_rslt = job_result[job_name]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                elif job_name == "factor_quantile":
                    stats_rslt = job_result[job_name]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                elif job_name == "factor_quantile_without_scaling":
                    stats_rslt = job_result[job_name]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                else:
                    assert False, "  error::>>factor_perf_tests>>density_stats:{0} is unknown!".format(job_name)
            elif job_cate == "decay_pnl_job":
                if job_name == "ic_decay":
                    stats_rslt = job_result[job_name]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["decay_pnl_bm_index"] = stats_attributes["decay_pnl_bm_index"]
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                else:
                    assert False, "  error::>>factor_perf_tests>>decay_stats:{0} is unknown!".format(job_name)
            elif job_cate == "decay_group_job":
                if job_name == "group_ret_decay":
                    stats_rslt = job_result[job_name]
                    stats_rslt["group"] = stats_rslt["group"].astype(str)
                    group_names = list(stats_rslt["group"].unique())
                    stats_rslt = stats_rslt.set_index(["factor_name", "lag_type", "group"]
                                                      )["stats_value"].unstack().reset_index(drop=False)
                    stats_rslt = stats_rslt[["factor_name", "lag_type"] + group_names].copy()
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["decay_grp_bm_index"] = stats_attributes["decay_pnl_bm_index"]
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                elif job_name == "group_ret_ir_decay":
                    stats_rslt = job_result[job_name]
                    stats_rslt["group"] = stats_rslt["group"].astype(str)
                    group_names = list(stats_rslt["group"].unique())
                    stats_rslt = stats_rslt.set_index(["factor_name", "lag_type", "group"]
                                                      )["stats_value"].unstack().reset_index(drop=False)
                    stats_rslt = stats_rslt[["factor_name", "lag_type"] + group_names].copy()
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["decay_grp_bm_index"] = stats_attributes["decay_pnl_bm_index"]
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                else:
                    assert False, "  error::>>factor_perf_tests>>decay_stats:{0} is unknown!".format(job_name)
            elif job_cate == "industry_pnl_job":
                if job_name == "industry_ic":
                    stats_rslt = job_result[job_name]
                    stats_rslt["start_date"] = stats_attributes["factor_start_date"]
                    stats_rslt["end_date"] = stats_attributes["ret_end_date"]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
            elif job_cate == "barra_job":
                if job_name == "barra_corr":
                    stats_rslt = job_result[job_name]
                    stats_rslt["factor_short_name"] = factor_shrt_name
                    stats_rslt["preprocess_method"] = preprocess_method
                    stats_rslt["freq"] = freq
                    stats_rslt["test_scope"] = ",".join(stats_attributes["test_scope"])
                else:
                    assert False, "  error::>>factor_perf_tests>>density_stats:{0} is unknown!".format(job_name)
            else:
                assert False, "  error::>>factor_perf_tests>>job_cate:{0} is unknown!".format(job_cate)
            stats_rslt['factor_name'] = stats_rslt['factor_name'].str.replace('-', '_')
            stats_rslt = stats_rslt[
                pd.Index(['factor_name', 'factor_short_name', 'preprocess_method', 'freq', 'test_scope']).append(
                    stats_rslt.columns.drop(['factor_name', 'factor_short_name', 'preprocess_method', 'freq', 'test_scope'])
                )]
            stats_rslt.to_excel(writer, sheet_name=job_name, index=False)
    return des_file_path


def _save_figures_in_pdf(root_save_path, stats_attributes, figures):
    factor_name = stats_attributes["factor_name"][0]
    pdf_name = "{0}@{1}@{2}.pdf".format(factor_name.replace('-', '_'), stats_attributes["factor_start_date"],
                                        stats_attributes["ret_end_date"])
    pdf_save_path = os.path.sep.join([root_save_path, stats_attributes["stock_universe_type"], factor_name.split('-')[0]])
    if not os.path.exists(pdf_save_path):
        os.makedirs(pdf_save_path)
    file_path = os.path.join(pdf_save_path, pdf_name)
    with PdfPages(file_path) as pdf:
        for fig in figures:
            pdf.savefig(fig[1])


def save_jobs(rslt, figures, save_config):
    stats_attributes = rslt["stats_attributes"]
    assert stats_attributes["factor_num"] == 1, \
        "  error::>>factor_perf_tests>>api>>Currently, only single factors are supported!"
    _save_figures_in_pdf(save_config["result_save_path"], stats_attributes, figures)
    excel_rslt_file = _save_job_results(save_config["result_save_path"], rslt,  save_config["save_job_list"],
                                        stats_attributes)
    return excel_rslt_file