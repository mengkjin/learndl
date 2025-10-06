import inspect
from functools import wraps

from ..fmp_top import plot as top_plot

def update_default_title_prefix(func):
    sig = inspect.signature(func)
    assert 'title_prefix' in sig.parameters , f'{func.__name__} must have a title_prefix parameter'
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'title_prefix' not in kwargs:
            kwargs['title_prefix'] = 'Screening Port'
        return func(*args, **kwargs)
    return wrapper

plot_top_frontface = update_default_title_prefix(top_plot.plot_top_frontface)
plot_top_perf_curve = update_default_title_prefix(top_plot.plot_top_perf_curve)
plot_top_perf_excess = update_default_title_prefix(top_plot.plot_top_perf_excess)
plot_top_perf_drawdown = update_default_title_prefix(top_plot.plot_top_perf_drawdown)
plot_top_perf_excess_drawdown = update_default_title_prefix(top_plot.plot_top_perf_excess_drawdown)
plot_top_perf_year = update_default_title_prefix(top_plot.plot_top_perf_year)
plot_top_perf_month = update_default_title_prefix(top_plot.plot_top_perf_month)
plot_top_exp_style = update_default_title_prefix(top_plot.plot_top_exp_style)
plot_top_exp_indus = update_default_title_prefix(top_plot.plot_top_exp_indus)
plot_top_attrib_source = update_default_title_prefix(top_plot.plot_top_attrib_source)
plot_top_attrib_style = update_default_title_prefix(top_plot.plot_top_attrib_style)