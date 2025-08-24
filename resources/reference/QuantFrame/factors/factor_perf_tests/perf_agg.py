import os
import pandas as pd


def load_single_factor_rslt(factor_test_rslt_excel, sheet_name_list=None):
    rtn = pd.read_excel(factor_test_rslt_excel, sheet_name=sheet_name_list)
    return rtn


def _agg_base_info_rslt(ic_rslts, group_perf_df, long_short_df):
    base_info_data = pd.Series(dtype="object")
    base_info_data["factor_name"] = ic_rslts["factor_name"]
    base_info_data['factor_short_name'] = ic_rslts['factor_short_name']
    base_info_data["start_date"] = ic_rslts["start_date"]
    base_info_data["end_date"] = ic_rslts["end_date"]
    base_info_data["freq"] = ic_rslts["freq"]
    base_info_data["preprocess_method"] = ic_rslts["preprocess_method"]
    base_info_data["test_scope"] = ic_rslts["test_scope"]
    base_info_data["pnl_bm_index"] = ic_rslts["pnl_bm_index"]
    base_info_data["group_bm_index"] = group_perf_df["group_bm_index"][0]
    #
    base_info_data["ic_mean"] = ic_rslts["ic_mean"]
    base_info_data["ic_ir"] = ic_rslts["ic_ir"]
    #
    grp_ret_mean = group_perf_df.filter(regex="group[0-9]+", axis=1).mean()
    base_info_data = pd.concat((base_info_data, grp_ret_mean), axis=0)
    #
    base_info_data["long"] = long_short_df["long"].mean()
    base_info_data["short"] = long_short_df["short"].mean()
    base_info_data["long_short"] = long_short_df["long_short"].mean()
    base_info_data = base_info_data.to_frame().T.set_index("factor_name")
    return base_info_data


def agg_single_factor_rslt(factor_test_rslt_excel):
    hist_factor_rslt = load_single_factor_rslt(factor_test_rslt_excel,
                                               sheet_name_list=["long_short_curve", "ic_curve", "ic_results",
                                                                "group_perf_curve"])
    ic_rslts = hist_factor_rslt["ic_results"].iloc[0, :]
    long_short_df = hist_factor_rslt["long_short_curve"].set_index(["CalcDate"])
    group_perf_df = hist_factor_rslt["group_perf_curve"].set_index(["CalcDate"])
    #
    base_info_data = _agg_base_info_rslt(ic_rslts, group_perf_df, long_short_df)
    #
    ic_df = hist_factor_rslt["ic_curve"].set_index(["CalcDate"])
    ic_info_data = ic_df["ic"].to_frame(name=ic_df["factor_name"][0])
    ic_info_data.index.rename("date", inplace=True)
    #
    ret_info_data = long_short_df["long"].to_frame(name=long_short_df["factor_name"][0])
    ret_info_data.index.rename("date", inplace=True)
    rtn = {"base_info": base_info_data, "ic_info": ic_info_data, "ret_info": ret_info_data}
    return rtn


def save_agg_job_rslts(factor_test_rslt_excel, agg_result_save_path, file_name):
    if not os.path.exists(agg_result_save_path):
        os.makedirs(agg_result_save_path)
    new_agg_rslt = agg_single_factor_rslt(factor_test_rslt_excel)
    #
    file_path = os.path.join(agg_result_save_path, "{0}.xlsx".format(file_name))
    if os.path.exists(file_path):
        all_agg_rslts = dict()
        hist_agg_rslts = pd.read_excel(file_path, index_col=0, sheet_name=None)
        all_agg_rslts["base_info"] = pd.concat((hist_agg_rslts["base_info"], new_agg_rslt["base_info"]), axis=0)
        #
        assert new_agg_rslt["ic_info"].index.equals(hist_agg_rslts["ic_info"].index), \
            "  errors::>>perf_agg>> hist data dates not equal new data!"
        exist_factor = list(set(hist_agg_rslts["ic_info"].columns) & set(new_agg_rslt["ic_info"].columns))
        all_agg_rslts["ic_info"] = pd.concat((hist_agg_rslts["ic_info"].drop(columns=exist_factor),
                                              new_agg_rslt["ic_info"]), axis=1)
        all_agg_rslts["ret_info"] = pd.concat((hist_agg_rslts["ret_info"].drop(columns=exist_factor),
                                               new_agg_rslt["ret_info"]), axis=1)
    else:
        all_agg_rslts = new_agg_rslt.copy()
    #
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name in ["base_info", "ic_info", "ret_info"]:
            all_agg_rslts[sheet_name].to_excel(writer, sheet_name=sheet_name, index=True)