import pandas as pd
import seaborn as sns
from events_system.calendar_util import CALENDAR_UTIL
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# for Chinese
sns.set(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def to_percent(temp, position):
  return '%.1f'%(100*temp) + '%'

def to_d2f(temp, position):
  return '%.2f'%temp

def plot_lag_perf_curve(results):
    lag_perf = results.set_index(["CalcDate", "lag_num"])["ex_ret"].unstack()
    lag_perf = lag_perf.dropna()
    assert set(lag_perf.index).issubset(CALENDAR_UTIL.get_ranged_trading_dates(lag_perf.index[0], lag_perf.index[-1]))
    lag_perf = lag_perf[lag_perf.index[1]:].copy()
    results = lag_perf.cumsum()
    results.columns.rename("lag_prd", inplace=True)
    results.index.rename("CalcDate", inplace=True)
    results = results.stack().rename("cum_perf").reset_index(drop=False)
    results = results.assign(CalcDate=pd.to_datetime(results['CalcDate'], format='%Y-%m-%d'))
    fig = plt.figure(figsize=(16, 7))
    #
    ax = sns.lineplot(x="CalcDate", y="cum_perf", hue="lag_prd", data=results)
    #ax.set_title("组合累计超额收益率")
    fig.suptitle("组合累计超额收益率", fontsize=12, y=0.95)
    plt.grid()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    return fig


def plot_year_perf_rslts(results):
    year_results = results[0].set_index(["年份"]).T
    year_results.loc[["绝对收益", "基准收益", "超额收益", "相对最大回撤", "最大回撤", "跟踪误差"]] = \
        year_results.loc[["绝对收益", "基准收益", "超额收益", "相对最大回撤", "最大回撤", "跟踪误差"]].applymap(lambda x: '{:.2%}'.format(x))
    year_results.loc[["信息比率", "相对收益回撤比", "换手率"]] = year_results.loc[["信息比率", "相对收益回撤比", "换手率"]]. \
        applymap(lambda x: '{:.3f}'.format(x) if not pd.isnull(x) else None)
    year_results.loc[["最大回撤起始日期", "最大回撤结束日期"]] = year_results.loc[["最大回撤起始日期", "最大回撤结束日期"]].applymap(lambda x: x.replace("-", ""))
    fig = plt.figure(figsize=(16, 7))
    ax = plt.table(cellText=year_results.values,
                   colLabels=year_results.columns, rowLabels=year_results.index,
                   rowLoc="center", loc="center")
    ax.scale(1, 2)
    plt.gcf().subplots_adjust(left=0.2)
    plt.suptitle("年度收益分析", fontsize=12, y=0.9)
    #plt.title("年度收益分析")
    plt.axis("off")
    return fig


def plot_prefix(all_stats):
    fig = plt.figure(figsize=(16, 7))
    ax = plt.table(cellText=all_stats.values,
                   colLabels=all_stats.columns, rowLabels=all_stats.index,
                   rowLoc="center", loc="center")
    ax.scale(1, 2)
    plt.gcf().subplots_adjust(left=0.3)
    plt.suptitle("封面信息", fontsize=12, y=0.9)
    plt.axis("off")
    return fig


def plot_test_info(results):
    year_results = results["ret_analysis_job"][0].set_index(["年份"]).loc["全样本"].copy()
    year_results[["绝对收益", "基准收益", "超额收益", "跟踪误差"]] = \
        year_results[["绝对收益", "基准收益", "超额收益", "跟踪误差"]].apply(lambda x: '{:.2%}'.format(x))
    year_results[["信息比率", "换手率"]] = year_results.loc[["信息比率", "换手率"]]. \
        apply(lambda x: '{:.3f}'.format(x) if not pd.isnull(x) else None)
    perf_stats = year_results[["绝对收益", "基准收益", "超额收益", "跟踪误差", "信息比率", "换手率"]].to_frame("Value")
    perf_stats["Type"] = "performance"
    #
    stats_attributes = results["stats_attributes"]
    param_stats = [("bm_index", stats_attributes["bm_index"].split(":")[1].split("@")[0], "bm_index"),
                   ("freq_type", stats_attributes["freq_type"], "freq_type"),
                   ("init_cash", stats_attributes["init_cash"], "back_test"),
                   ("trading_fee", stats_attributes["accnt_cfg"]["trading_fee"], "back_test")]
    param_stats = pd.DataFrame(param_stats, columns=["param_name", "Value", "Type"]).set_index("param_name")
    #
    all_stats = pd.concat((param_stats, perf_stats), axis=0)
    fig = plt.figure(figsize=(16, 7))
    ax = plt.table(cellText=all_stats.values,
                   colLabels=all_stats.columns, rowLabels=all_stats.index,
                   rowLoc="center", loc="center")
    ax.scale(1, 2)
    plt.gcf().subplots_adjust(left=0.2)
    #plt.title("组合相关信息")
    plt.suptitle("组合相关信息", fontsize=12, y=0.9)
    plt.axis("off")
    return fig


def plot_ret_attribution_curve(port_dcmp_ret):
    port_dcmp_ret = port_dcmp_ret.dropna(axis=0, how="all")
    port_dcmp_ret.columns.name = "sector_name"
    port_dcmp_cum_ret = port_dcmp_ret.cumsum()
    port_dcmp_cum_ret = port_dcmp_cum_ret.stack().rename("sector_ret").reset_index(drop=False)
    port_dcmp_cum_ret["CalcDate"] = pd.to_datetime(port_dcmp_cum_ret["CalcDate"])
    fig = plt.figure(figsize=(16, 7))
    custom_colors = ['black', 'purple', 'orange', 'green', 'cornflowerblue', 'red'] 
    ax = sns.lineplot(x="CalcDate", y="sector_ret", hue="sector_name", data=port_dcmp_cum_ret, palette=custom_colors)
    #ax.set_title("组合收益分解")
    fig.suptitle("组合收益分解", fontsize=12, y=0.95)
    plt.grid()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    return fig


def plot_style_attribution_curve(port_dcmp_ret):
    port_dcmp_ret = port_dcmp_ret.dropna(axis=0, how="all")
    port_dcmp_ret.columns.name = "sector_name"
    port_dcmp_cum_ret = port_dcmp_ret.cumsum()
    port_dcmp_cum_ret.index = pd.to_datetime(port_dcmp_cum_ret.index)
    #
    factor_list = port_dcmp_cum_ret.columns.tolist()
    factor_list.append('Total')
    port_dcmp_cum_ret['Total'] = port_dcmp_cum_ret.sum(axis = 1)
    #
    lay_out = (4, 4)
    row_num, col_num = lay_out[0], lay_out[1]
    fig_num = len(factor_list)
    assert len(factor_list) <= row_num * col_num
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 7))
    for i in range(fig_num):
        ax = axs[int(i / col_num), i % col_num]
        f_nm = factor_list[i]
        ax.plot(port_dcmp_cum_ret[f_nm].index, port_dcmp_cum_ret[f_nm], label=f_nm)
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.grid()
        ax.legend()
    '''
    for i in range(fig_num, row_num * col_num):
        ax = axs[int(i / col_num), i % col_num]
        ax.plot(port_dcmp_cum_ret.index.tolist(), np.full((port_dcmp_cum_ret.shape[0],), 0.0))
        ax.grid()
        ax.set_ylim(0, 0.001)
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    '''
    fig.suptitle("风格因子累计贡献", fontsize=12, y=0.95)
    fig.autofmt_xdate()
    return fig

def plot_style_exposure_curve(port_sylte_risk, port_constituent_ratio):
    port_sylte_risk.index = pd.to_datetime(port_sylte_risk.index)
    port_constituent_ratio.index = pd.to_datetime(port_constituent_ratio.index)
    factor_list = port_sylte_risk.columns.tolist()
    #
    lay_out = (4, 4)
    row_num, col_num = lay_out[0], lay_out[1]
    fig_num = len(factor_list)
    assert len(factor_list) <= row_num * col_num
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 7))
    for i in range(fig_num):
        ax = axs[int(i / col_num), i % col_num]
        f_nm = factor_list[i]
        ax.plot(port_sylte_risk[f_nm].index, port_sylte_risk[f_nm], label=f_nm)
        ax.yaxis.set_major_formatter(FuncFormatter(to_d2f))
        ax.grid()
        ax.legend()
    for i in range(fig_num, row_num * col_num):
        ax = axs[int(i / col_num), i % col_num]
        ax.plot(port_constituent_ratio.index, port_constituent_ratio['weight'], label="con_ratio")
        ax.yaxis.set_major_formatter(FuncFormatter(to_d2f))
        ax.grid()
        ax.legend()
    fig.suptitle("风格因子暴露", fontsize=12, y=0.95)
    fig.autofmt_xdate()
    return fig

def plot_industry_bias_curve(port_industry_bias):
    port_industry_bias.index = pd.to_datetime(port_industry_bias.index)
    industry_list = port_industry_bias.columns.tolist()
    #
    lay_out = (6, 5)
    row_num, col_num = lay_out[0], lay_out[1]
    fig_num = len(industry_list)
    assert len(industry_list) <= row_num * col_num
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 7))
    for i in range(fig_num):
        ax = axs[int(i / col_num), i % col_num]
        f_nm = industry_list[i]
        ax.tick_params(labelsize=9)
        ax.plot(port_industry_bias[f_nm].index, port_industry_bias[f_nm], label=f_nm)
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
        ax.grid()
        ax.legend()
    fig.suptitle("行业偏离", fontsize=12, y=0.95)
    fig.autofmt_xdate()
    return fig
