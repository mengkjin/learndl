import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from packaging import version
from typing import Any , Callable , Literal , Optional

# for Chinese
sns.set_theme(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_SEABORN_VERSION = version.Version(getattr(sns , '__version__')) > version.Version('0.9.1')

def multi_factor_plot(func : Callable):
    def wrapper(df : pd.DataFrame , *args , factor_name : Optional[str] = None , **kwargs) -> Any | list[Any]:
        if factor_name is None:
            factor_list = df['factor_name'].unique()
            return [func(df , *args , factor_name , **kwargs) for factor_name in factor_list]
        else:
            return func(df , *args , factor_name , **kwargs)
    return wrapper

def plot_head(df : pd.DataFrame , factor_name : Optional[str] = None) -> tuple[pd.DataFrame , Figure]:
    assert factor_name is not None and (factor_name in df['factor_name'].values) , factor_name
    df = df[df['factor_name'] == factor_name].drop(columns=['factor_name'])
    fig = plt.figure(figsize=(16, 7))
    return df , fig

def plot_end(title : Optional[str] = None):
    if isinstance(title , str): plt.title(title)
    plt.grid()
    plt.xticks(rotation=45)  

def pct_fmt(temp : float, position : int = 2): return f'{np.format_float_positional(temp * 100 , position)}%'

@multi_factor_plot
def plot_decay_ic(df : pd.DataFrame , factor_name : Optional[str] = None):
    df , fig = plot_head(df , factor_name)

    bars = plt.bar(df['lag_type'].astype('category').cat.codes  , df['ic_mean'], label=factor_name)  

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2 , height ,
                f'{height:.4f}' , ha = 'center' , va = 'top' if height < 0 else 'bottom')

    plot_end(title = f'Average IC Decay for [{factor_name}]')
    return fig

@multi_factor_plot
def plot_decay_grp_perf(df : pd.DataFrame , stat_type : Literal['ret' , 'ir'] = 'ret',  factor_name : Optional[str] = None):
    assert factor_name is not None and (factor_name in df['factor_name'].values) , factor_name
    df = df[df['factor_name'] == factor_name].drop(columns=['factor_name'])
    fig = plt.figure(figsize=(16, 7))

    grp_perf = df[(df['stats_name'] == f'decay_grp_{stat_type}')]
    group_num = grp_perf['group'].nunique()

    if CURRENT_SEABORN_VERSION:
        ax = sns.barplot(x='lag_type', y='stats_value', hue='group', data=grp_perf,
                        palette=sns.diverging_palette(140, 10, sep=10, n=group_num))
    else:
        ax = sns.barplot(x='lag_type', y='stats_value', hue='group', data=grp_perf)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
    ax.set_title(f'Groups {stat_type.upper()} Decay for [{factor_name}]') 
    ax.legend(loc='upper left')

    plt.grid()
    plt.xticks(rotation=45)  
    return fig
    
@multi_factor_plot
def plot_grp_perf(df : pd.DataFrame , factor_name : Optional[str] = None):
    assert factor_name is not None and (factor_name in df['factor_name'].values) , factor_name
    df = df[df['factor_name'] == factor_name].drop(columns=['factor_name'])
    fig = plt.figure(figsize=(16, 7))

    grp_cum_ret = df.set_index(['date', 'group']).groupby('group' , observed=False)['group_ret'].\
        cumsum().rename('cum_ret').reset_index(drop=False)
    grp_cum_ret = grp_cum_ret.assign(date=pd.to_datetime(grp_cum_ret['date'].astype(str), format='%Y%m%d'))

    if CURRENT_SEABORN_VERSION:
        ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=grp_cum_ret,
                            palette=sns.diverging_palette(140, 10, sep=10, n=grp_cum_ret['group'].nunique()))
    else:
        ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=grp_cum_ret)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.set_title(f'Group CumReturn for [{factor_name}]')

    plt.grid()
    plt.xticks(rotation=45)  
    return fig
    
@multi_factor_plot
def plot_style_corr(df : pd.DataFrame , factor_name : Optional[str] = None):
    assert factor_name is not None and (factor_name in df['factor_name'].values) , factor_name
    df = df[df['factor_name'] == factor_name].drop(columns=['factor_name'])
    fig = plt.figure(figsize=(16, 7))

    style_corr = df.set_index(['date'])
    style_corr.columns.rename('style_factor', inplace=True)
    style_corr = style_corr.stack().rename('factor_corr').reset_index(drop=False) # type: ignore

    ax = sns.boxplot(x='style_factor', y='factor_corr', data=style_corr, width=0.3)
    ax.set_title(f'Correlation with Risk Style Factors for [{factor_name}]')

    plt.grid()
    plt.xticks(rotation=45)  
    return fig

@multi_factor_plot
def plot_distribution(df : pd.DataFrame , factor_name : Optional[str] = None):
    df , fig = plot_head(df , factor_name)

    density_info = df
    assert not density_info['date'].duplicated().any()
    #
    col_num = 3
    row_num = int(np.ceil(len(density_info) / col_num))
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 7))
    for i in range(len(density_info)):
        day_density_info = density_info.iloc[i]
        bins = day_density_info['hist_bins']
        cnts = day_density_info['hist_cnts']
        cnts = cnts / cnts.sum()

        ax = axs[int(i / col_num), i % col_num]
        ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), facecolor='red')

    fig.suptitle(f'Factor Distribution for {factor_name}')
    return fig

@multi_factor_plot
def plot_factor_qtile(df : pd.DataFrame , factor_name : Optional[str] = None):
    df , fig = plot_head(df , factor_name)

    df.columns.rename('quantile_name', inplace=True)
    df = df.set_index('date').stack().rename('quantile_value').reset_index(drop=False) # type: ignore
    df = df.assign(date=pd.to_datetime(df['date'].astype(str)), format='%Y%m%d')

    ax = sns.lineplot(x='date', y='quantile_value', hue='quantile_name', data=df)
    plt.grid()
    ax.set_title(f'Factor Quantile for [{factor_name}]')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    plot_end()
    return fig