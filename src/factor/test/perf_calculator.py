from factor_stats.pnl_perf import calc_pnl_perf
from factor_stats.ic_perf import calc_ic_perf, evaluate_ic_yearly, evaluate_ic
from factor_stats.density_est import calc_density_info, calc_factor_qtile, calc_factor_qtile_without_scaling
from factor_stats.grouping import calc_grp_ret, evaluate_top_perf_yearly
from factor_stats.alpha_decay import calc_decay_pnl, calc_decay_grp_perf
from factor_stats.industry_alpha import calc_industry_ic
from factor_stats.barra_corr import calc_barra_corr
from datetime import datetime
from factor_stats.fcst_stats_utils.data_preparation import init_data_center
from ashare_stkpool.api import remove_outpool_data
import pandas as pd
from industry.api import load_industry_data


def calc_pnl_job(factor_val_df, ret_type, price_type, ecd, stats_list, pnl_bm_index, ic_type, freq_type):
    rtn = dict()
    for stats in stats_list:
        if stats == "long_short_curve":
            rtn[stats] = calc_pnl_perf(factor_val_df, ret_type, price_type, ecd, pnl_bm_index, ["long_short", "long", "short"])[0]
        elif stats == "ic_curve":
            ic_data = calc_ic_perf(factor_val_df, ret_type, price_type, ecd, pnl_bm_index, ic_type)
            ic_results = evaluate_ic(ic_data, freq_type)
            ic_yearly_results = evaluate_ic_yearly(ic_data, freq_type)
            rtn[stats] = (ic_data, ic_results, ic_yearly_results)
        else:
            assert False, "  error::>>factor_perf_tests>>pnl_stats:{0} is unknown!".format(stats)
    return rtn


def calc_density_job(factor_val_df, stats_list, factor_sampling_num, hist_bins):
    rtn = dict()
    for stats in stats_list:
        if stats == "factor_distribution":
            rtn[stats] = calc_density_info(factor_val_df, factor_sampling_num, hist_bins)
        elif stats == "factor_quantile":
            rtn[stats] = calc_factor_qtile(factor_val_df)
        elif stats == "factor_quantile_without_scaling":
            rtn[stats] = calc_factor_qtile_without_scaling(factor_val_df)
        else:
            assert False, "  error::>>factor_perf_tests>>density_stats:{0} is unknown!".format(stats)
    return rtn


def calc_group_job(factor_val_df, ret_type, price_type, ecd, stats_list, grp_bm_index, group_num, freq_type):
    rtn = dict()
    for stats in stats_list:
        if stats == "group_perf_curve":
            group_perf = calc_grp_ret(factor_val_df, ret_type, price_type, ecd, grp_bm_index, group_num)
            top_year_perf = evaluate_top_perf_yearly(group_perf, freq_type)
            rtn[stats] = (group_perf, top_year_perf)
        else:
            assert False, "  error::>>factor_perf_tests>>group_stats:{0} is unknown!".format(stats)
    return rtn


def calc_decay_grp_job(factor_val_df, ret_type, freq_type, ecd, stats_list, bm_index_nm, group_num, ret_range_type, price_type, lag_num):
    rtn = dict()
    decay_grp_perf = calc_decay_grp_perf(factor_val_df, ret_type, freq_type, ecd, bm_index_nm, group_num, ret_range_type, price_type, lag_num)
    for stats in stats_list:
        if stats == "group_ret_decay":
            rtn[stats] = decay_grp_perf[decay_grp_perf["stats_name"] == "decay_grp_ret"].copy()
        elif stats == "group_ret_ir_decay":
            rtn[stats] = decay_grp_perf[decay_grp_perf["stats_name"] == "decay_grp_ir"].copy()
        else:
            assert False, "  error::>>factor_perf_tests>>decay_stats:{0} is unknown!".format(stats)
    return rtn


def calc_decay_pnl_job(factor_val_df, ret_type, price_type, ecd, stats_list, bm_index_nm, ret_range_type, lag_num, ic_type):
    rtn = dict()
    for stats in stats_list:
        if stats == "ic_decay":
            rtn[stats] = calc_decay_pnl(factor_val_df, ret_type, ecd, bm_index_nm, ret_range_type, price_type,
                                        ic_type, lag_num)
        else:
            assert False, "  error::>>factor_perf_tests>>decay_stats:{0} is unknown!".format(stats)
    return rtn


def calc_industry_pnl_job(root_path, factor_val_df, ret_type, price_type, ecd, stats_list, industry_type, bm_index_nm, ic_type):
    rtn = dict()
    for stats in stats_list:
        if stats == "industry_ic":
            rtn[stats] = calc_industry_ic(root_path, factor_val_df, ret_type, price_type, industry_type, ecd, bm_index_nm, ic_type)
        else:
            assert False, "  error::>>factor_perf_tests>>industry_pnl_stats:{0} is unknown!".format(stats)
    return rtn


def calc_barra_job(root_path, factor_val_df, stats_list, barra_type):
    rtn = dict()
    for stats in stats_list:
        if stats == "barra_corr":
            rtn[stats] = calc_barra_corr(root_path, factor_val_df, barra_type)
        else:
            assert False, "  error::>>factor_perf_tests>>industry_pnl_stats:{0} is unknown!".format(stats)
    return rtn


def _filter_stock_by_scope(root_path, factor_val_df, configs, test_scope):
    assert isinstance(test_scope, list)
    factor_val_df = remove_outpool_data(root_path, factor_val_df, pool_type=configs['stock_universe_type'])
    if 'all' in test_scope:
        rtn = factor_val_df.copy()
    else:
        print("  warning::factor_perf_tests>>api>>filter_by_scope>>deal with industry only.")
        industry_type = configs["industry_type"]
        scd, ecd = factor_val_df["CalcDate"].min(), factor_val_df["CalcDate"].max()
        industry_data = load_industry_data(root_path, scd, ecd, industry_type, False)
        all_ind_list = industry_data[industry_type].unique()  # TODO: the way to get industry list should be impled as by a util.
        assert industry_type not in factor_val_df.columns
        rtn = pd.merge(factor_val_df, industry_data, on=["CalcDate", "Code"], how="left")
        unknown_industry = set(test_scope).difference(set(all_ind_list))
        assert not unknown_industry, "  error::factor_perf_tests>>perf_showing>>unknown industries found:{0}.".format(','.join(list(unknown_industry)))
        rtn = pd.merge(rtn, pd.DataFrame(test_scope, columns=[industry_type]), how='inner', on=[industry_type]).drop(columns=[industry_type])
    return rtn


def verify_attributes(factor_val_df, ecd, configs, test_scope, freq_type):
    assert 'CalcDate' in factor_val_df.columns and 'Code' in factor_val_df.columns
    factor_val_df = factor_val_df.sort_values(by=['CalcDate', 'Code'])
    #
    rtn = dict()
    factor_col = factor_val_df.columns.drop(["CalcDate", "Code"])
    rtn["factor_num"] = len(factor_col)
    rtn["factor_name"] = factor_col.tolist()
    rtn["test_scope"] = test_scope
    rtn["factor_start_date"] = factor_val_df["CalcDate"].min()
    rtn["factor_end_date"] = factor_val_df["CalcDate"].max()
    rtn["ret_end_date"] = ecd
    rtn["test_date"] = datetime.now().strftime('%Y-%m-%d')
    rtn['freq_type'] = freq_type
    rtn.update(configs)
    return rtn


def calc_jobs(root_path, factor_val_df, ecd, configs, test_scope, freq_type):
    assert len(factor_val_df.columns) <= 2 + configs['max_factor']
    factor_val_df = _filter_stock_by_scope(root_path, factor_val_df, configs, test_scope)
    init_data_center(root_path)
    ret_type = configs["ret_type"]
    results = dict()
    results['stats_attributes'] = verify_attributes(factor_val_df, ecd, configs, test_scope, freq_type)
    if configs["pnl_job"] == 'on':
        results["pnl_job"] = calc_pnl_job(
            factor_val_df,
            ret_type,
            configs["price_type"],
            ecd,
            configs["pnl_stats_list"],
            configs["pnl_bm_index"],
            configs["pnl_ic_type"],
            freq_type)
    if configs["density_job"] == 'on':
        results["density_job"] = calc_density_job(
            factor_val_df,
            configs["density_stats_list"],
            configs["factor_sampling_num"],
            configs["hist_bins"])
    if configs["group_job"] == 'on':
        results["group_job"] = calc_group_job(
            factor_val_df,
            ret_type,
            configs["price_type"],
            ecd,
            configs["group_stats_list"],
            configs["group_bm_index"],
            configs["group_num"],
            freq_type)
    if configs["decay_pnl_job"] == 'on':
        results["decay_pnl_job"] = calc_decay_pnl_job(
            factor_val_df,
            ret_type,
            configs["price_type"],
            ecd,
            configs["decay_pnl_stats_list"],
            configs["decay_pnl_bm_index"],
            configs["pnl_ret_range_type"],
            configs["pnl_lag_num"],
            configs["decay_ic_type"])
    if configs["decay_group_job"] == 'on':
        results["decay_group_job"] = calc_decay_grp_job(
            factor_val_df,
            ret_type,
            freq_type,
            ecd,
            configs["decay_grp_stats_list"],
            configs["decay_grp_bm_index"],
            configs["decay_group_num"],
            configs["grp_ret_range_type"],
            configs["price_type"],
            configs["grp_lag_num"])
    if configs["industry_pnl_job"] == 'on':
        results["industry_pnl_job"] = calc_industry_pnl_job(
            root_path,
            factor_val_df,
            ret_type,
            configs["price_type"],
            ecd,
            configs["industry_pnl_stats_list"],
            configs["pnl_industry_type"],
            configs["industry_pnl_bm_index"],
            configs["industry_ic_type"])
    if configs["barra_job"] == 'on':
        results["barra_job"] = calc_barra_job(
            root_path,
            factor_val_df,
            configs["barra_stats_list"],
            configs["barra_type"])
    return results
