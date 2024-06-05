from .plot_util import plot_lag_perf_curve, plot_year_perf_rslts, plot_test_info, plot_prefix,\
    plot_ret_attribution_curve, plot_style_attribution_curve, plot_style_exposure_curve, plot_industry_bias_curve


def plot_jobs(job_rslts, configs, prefix_data):
    rtn = []
    if prefix_data is not None:
        rtn.append(("prefix", plot_prefix(prefix_data)))
    for show_job_name in configs:
        if show_job_name == "plot_lag_perf_curve":
            job_result = job_rslts["lag_analysis_job"]
            figure = plot_lag_perf_curve(job_result)
        elif show_job_name == "plot_year_perf_rslts":
            job_result = job_rslts["ret_analysis_job"]
            figure = plot_year_perf_rslts(job_result)
        elif show_job_name == "plot_test_info":
            figure = plot_test_info(job_rslts)
        elif show_job_name == "ret_attribution_curve":
            result = job_rslts["ret_attribution_job"][0]
            figure = plot_ret_attribution_curve(result)
        elif show_job_name == "style_attribution_curve":
            result = job_rslts["ret_attribution_job"][1]
            figure = plot_style_attribution_curve(result)
        elif show_job_name == "style_exposure_curve":
            #result = job_rslts["risk_analysis_job"][0]
            figure = plot_style_exposure_curve(job_rslts["risk_analysis_job"][0], job_rslts["risk_analysis_job"][3])     
        elif show_job_name == "industry_bias_curve":
            result = job_rslts["risk_analysis_job"][1]
            figure = plot_industry_bias_curve(result)  
        else:
            assert False, "  error::>>fmp_perf_tests>>show_job_name:{0} is unknown!".format(show_job_name)
        rtn.append((show_job_name, figure))
    return rtn