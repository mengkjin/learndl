from .perf_calculator import calc_jobs
from .perf_showing import plot_jobs
from .perf_saver import save_jobs
from .perf_agg import save_agg_job_rslts


def _get_configs(configs):
    calc_config = configs["CALC_CONFIG"]
    show_config = configs["SHOW_CONFIG"]
    save_config = configs["SAVE_CONFIG"]
    return calc_config, show_config, save_config


def test_factor(root_path, factor_val_df, freq_type, ecd, test_scope, configs):
    calc_config, show_config, save_config = _get_configs(configs)
    #
    stats_rslts = calc_jobs(root_path, factor_val_df, ecd, calc_config, test_scope, freq_type)
    #
    figures = plot_jobs(stats_rslts, show_config)
    #
    test_rslt_excel = save_jobs(stats_rslts, figures, save_config)
    return test_rslt_excel


def agg_factor(factor_test_rslt_excel, agg_result_save_path, agg_file_name):
    save_agg_job_rslts(factor_test_rslt_excel, agg_result_save_path, agg_file_name)












