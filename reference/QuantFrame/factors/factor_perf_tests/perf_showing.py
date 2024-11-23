from .plot_util import plot_factor_distribution, plot_decay_ic, plot_group_perf, plot_decay_grp_ret,\
    plot_decay_grp_ret_ir, plot_ic_curve, plot_long_short, plot_industry_ic, plot_factor_quantile, plot_barra_corr, \
    plot_factor_quantile_without_scaling, plot_ic_year_rslts, plot_top_perf_year_rslts


def plot_jobs(job_rslts, configs):
    stats_attributes = job_rslts["stats_attributes"]
    assert stats_attributes["factor_num"] == 1, \
        "  error::>>factor_perf_tests>>api>>Currently, only single factors are supported!"
    rtn = []
    for show_job_name, calc_job_cate, calc_job_name in configs:
        job_result = job_rslts[calc_job_cate][calc_job_name]
        if show_job_name == "plot_long_short":
            figure = plot_long_short(job_result)
        elif show_job_name == "plot_ic_curve":
            figure = plot_ic_curve(job_result)
        elif show_job_name == "plot_ic_year_rslts":
            figure = plot_ic_year_rslts(job_result)
        elif show_job_name == "plot_group_perf":
            figure = plot_group_perf(job_result)
        elif show_job_name == "plot_top_perf_year_rslts":
            figure = plot_top_perf_year_rslts(job_result)
        elif show_job_name == "plot_factor_distribution":
            figure = plot_factor_distribution(job_result)
        elif show_job_name == "plot_factor_quantile":
            figure = plot_factor_quantile(job_result)
        elif show_job_name == "plot_factor_quantile_without_scaling":
            figure = plot_factor_quantile_without_scaling(job_result)
        elif show_job_name == "plot_decay_ic":
            figure = plot_decay_ic(job_result)
        elif show_job_name == "plot_decay_grp_ret":
            figure = plot_decay_grp_ret(job_result)
        elif show_job_name == "plot_decay_grp_ret_ir":
            figure = plot_decay_grp_ret_ir(job_result)
        elif show_job_name == "plot_industry_ic":
            figure = plot_industry_ic(job_result)
        elif show_job_name == "plot_barra_corr":
            figure = plot_barra_corr(job_result)
        else:
            assert False, "  error::>>factor_perf_tests>>show_job_name:{0} is unknown!".format(show_job_name)
        rtn.append((show_job_name, figure))
    return rtn
