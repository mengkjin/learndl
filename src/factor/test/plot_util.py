import numpy as np
import pandas as pd
import seaborn as sns
from distutils.version import StrictVersion
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# for Chinese
sns.set_theme(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def to_percent(temp, position):
  return '%.2f'%(100*temp) + '%'


def plot_factor_distribution(density_info):
    factor_list = density_info['factor_name'].unique()
    factor_name = factor_list[0]
    factor_density_info = density_info[density_info['factor_name'] == factor_name].drop(columns=['factor_name'])
    assert not factor_density_info['CalcDate'].duplicated().any()
    #
    col_num = 3
    date_num = factor_density_info.shape[0]
    row_num = int(np.ceil(date_num / col_num))
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 7))
    for i in range(date_num):
        day_density_info = factor_density_info.iloc[i].copy()
        date = day_density_info['CalcDate']
        bins = day_density_info['hist_bins']
        cnts = day_density_info['hist_cnts']
        cnts = cnts / cnts.sum()
        #
        ax = axs[int(i / col_num), i % col_num]
        ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), facecolor='red')
        ax.set_title(date.replace('-', ''))
    fig.suptitle('因子分布抽样图')
    return fig

def plot_factor_quantile(factor_qtile_data):
    factor_qtile_data = factor_qtile_data.set_index(['CalcDate', 'factor_name'])
    factor_qtile_data.columns.rename('quantile_name', inplace=True)
    factor_qtile_data = factor_qtile_data.stack().rename('quantile_value').reset_index(drop=False)
    factor_list = factor_qtile_data['factor_name'].unique()
    factor_name = factor_list[0]
    factor_qtile_data = factor_qtile_data[factor_qtile_data['factor_name'] == factor_name].drop(columns=['factor_name'])
    factor_qtile_data = factor_qtile_data.assign(CalcDate=pd.to_datetime(factor_qtile_data['CalcDate'], format='%Y-%m-%d'))
    fig = plt.figure(figsize=(16, 7))
    # TODO: add norm distribution tick
    ax = sns.lineplot(x='CalcDate', y='quantile_value', hue='quantile_name', data=factor_qtile_data)
    ax.set_title('因子分位点时序变化')
    plt.grid()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    return fig

def plot_ic_curve(stats_rslts):
    ic_perf = stats_rslts[0]
    ic_cumsum = ic_perf.cumsum()
    fig = plt.figure(figsize=(16, 7))
    plt.plot(pd.to_datetime(ic_cumsum.index), ic_cumsum.values)
    plt.grid()
    plt.title('IC累加')
    return fig


def plot_long_short(long_short_perf):
    long_short_perf = long_short_perf.stack().reset_index(drop=False)
    factor_list = long_short_perf['factor_name'].unique()
    factor_name = factor_list[0]
    long_short_perf = long_short_perf[long_short_perf['factor_name'] == factor_name].drop(columns=['factor_name'])
    #
    long_short_perf.set_index('CalcDate', inplace=True)
    long_short_cumsum = long_short_perf.cumsum()
    fig = plt.figure(figsize=(16, 7))
    x = pd.to_datetime(long_short_cumsum.index)
    plt.plot(x, long_short_cumsum['short'] * (-1), label='-short')
    plt.plot(x, long_short_cumsum['long_short'], label='long-short')
    plt.plot(x, long_short_cumsum['long'], label='long')
    plt.grid()
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.title('多空累计收益')
    return fig


def plot_ic_year_rslts(results):
    ic_results = results[2]
    factor_list = ic_results['factor_name'].unique()
    factor_name = factor_list[0]
    ic_rslt = ic_results[ic_results['factor_name'] == factor_name].drop(columns=['factor_name', 'direction'])
    ic_rslt = ic_rslt.rename(
        columns={'year': '年份', 'ic_mean': 'IC均值', 'ic_std': 'IC标准差',
                 'ic_ir': 'ICIR', 'ic_abs_mean': '|IC|均值', 'ic_maxdown': 'IC最大回撤'}, errors='raise').set_index('年份').T
    ic_rslt.loc[['IC均值', 'IC标准差', 'ICIR', '|IC|均值', 'IC最大回撤']] = \
        ic_rslt.loc[['IC均值', 'IC标准差', 'ICIR', '|IC|均值', 'IC最大回撤']]. \
        applymap(lambda x: '{:.3f}'.format(x) if not pd.isnull(x) else None)
    fig = plt.figure(figsize=(16, 7))
    ax = plt.table(cellText=ic_rslt.values,
                   colLabels=ic_rslt.columns, rowLabels=ic_rslt.index,
                   rowLoc='center', loc='center')
    direction = ic_results['direction'].values[0]
    ax.scale(1, 2)
    plt.title('[年IC分析]:{0}'.format('反向' if direction < 0 else '正向'))
    plt.axis('off')
    return fig


def plot_top_perf_year_rslts(results):
    top_results = results[1]
    factor_list = top_results['factor_name'].unique()
    factor_name = factor_list[0]
    perf_rslt = top_results[top_results['factor_name'] == factor_name].drop(columns=['factor_name', 'group'])
    perf_rslt = perf_rslt.rename(
        columns={'year': '年份', 'year_ret': '超额收益', 'ret_std': '超额收益标准差',
                 'ir': '超额IR', 'ret_max_down': '超额最大回撤'}, errors='raise').set_index('年份').T
    perf_rslt.loc[['超额收益', '超额收益标准差', '超额最大回撤']] = \
        perf_rslt.loc[['超额收益', '超额收益标准差', '超额最大回撤']]. \
        applymap(lambda x: '{:.2%}'.format(x) if not pd.isnull(x) else None)
    perf_rslt.loc[['超额IR']] = perf_rslt.loc[['超额IR']]. \
        applymap(lambda x: '{:.3f}'.format(x) if not pd.isnull(x) else None)
    fig = plt.figure(figsize=(16, 7))
    ax = plt.table(cellText=perf_rslt.values,
                   colLabels=perf_rslt.columns, rowLabels=perf_rslt.index,
                   rowLoc='center', loc='center')
    group = top_results['group'].values[0]
    ax.scale(1, 2)
    plt.title('[年超额收益分析]:{0}'.format(str(group)))
    plt.axis('off')
    return fig


def plot_industry_ic(results):
    factor_list = results['factor_name'].unique()
    factor_name = factor_list[0]
    results = results[results['factor_name'] == factor_name].drop(columns=['factor_name'])
    results.sort_values(['ic_ir'], ascending=False, inplace=True)
    #
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax1.set_title('各行业IC表现')
    plt.bar(results['industry'], results['ic_ir'], color='burlywood', alpha=0.5)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(['IC_IR'], loc='upper center')
    plt.xticks(rotation='vertical')
    #
    ax2 = ax1.twinx()
    plt.plot(results['industry'], results['ic_mean'], color='darkred')
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    plt.subplots_adjust(bottom=0.3)
    ax2.legend(['IC均值（右轴）'], loc='upper right')
    return fig

