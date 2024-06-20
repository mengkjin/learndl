import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from typing import Any , Callable , Literal , Optional

from ..basic.plot import CURRENT_SEABORN_VERSION , pct_fmt , multi_factor_plot , plot_head , plot_tail , plot_table

@multi_factor_plot
def plot_decay_ic(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index('lag_type')
    bars = plt.bar(df.index  , df['ic_mean'], label=factor_name)  

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2 , height ,
                 f'{height:.4f}' , ha = 'center' , va = 'top' if height < 0 else 'bottom')

    plt.grid()
    plot_tail(f'Average IC Decay' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_decay_grp_perf(df : pd.DataFrame , stat_type : Literal['ret' , 'ir'] = 'ret',  factor_name : Optional[str] = None , 
                        benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df[(df['stats_name'] == f'decay_grp_{stat_type}')]

    if CURRENT_SEABORN_VERSION:
        ax = sns.barplot(x='lag_type', y='stats_value', hue='group', data=df,
                        palette=sns.diverging_palette(140, 10, sep=10, n=df['group'].nunique()))
    else:
        ax = sns.barplot(x='lag_type', y='stats_value', hue='group', data=df)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
    #ax.set_title(f'Groups {stat_type.upper()} Decay for [{factor_name}]') 
    ax.legend(loc='upper left')
    plt.grid()

    plot_tail(f'Groups {stat_type.upper()} Decay' , factor_name , benchmark , show , suptitle = False)
    return fig

def plot_decay_grp_perf_ir(df : pd.DataFrame , factor_name : Optional[str] = None , 
                           benchmark : Optional[str] = None , show = False):
    return plot_decay_grp_perf(df , 'ir' , factor_name , benchmark , show)

def plot_decay_grp_perf_ret(df : pd.DataFrame , factor_name : Optional[str] = None , 
                           benchmark : Optional[str] = None , show = False):
    return plot_decay_grp_perf(df , 'ret' , factor_name , benchmark , show)
    
@multi_factor_plot
def plot_grp_perf(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index(['date', 'group']).groupby('group' , observed=False)['group_ret'].\
        cumsum().rename('cum_ret').reset_index(drop=False)
    # df = df.assign(date=pd.to_datetime(df['date'].astype(str), format='%Y%m%d'))

    if CURRENT_SEABORN_VERSION:
        ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=df,
                          palette=sns.diverging_palette(140, 10, sep=10, n=df['group'].nunique()))
    else:
        ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.2%}'))
    # ax.set_title(f'Group CumReturn for [{factor_name}]')

    plt.grid()

    plot_tail(f'Group CumReturn' , factor_name , benchmark , show , suptitle = False)
    return fig
    
@multi_factor_plot
def plot_style_corr_box(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index(['date'])
    df.columns.rename('style_factor', inplace=True)
    df = df.stack().rename('factor_corr').reset_index(drop=False) # type: ignore

    ax = sns.boxplot(x='style_factor', y='factor_corr', data=df, width=0.3)
    plt.grid()
    plot_tail(f'Correlation with Risk Style Factors' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_style_corr(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index(['date'])
    for style in df.columns.tolist(): plt.plot(df.index , df[style], label=style)
    plt.grid()

    plot_tail(f'Correlation Curve with Risk Style Factors' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_distribution(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    assert not df['date'].duplicated().any()
    #
    col_num = 3
    row_num = int(np.ceil(len(df) / col_num))
    fig, axs = plt.subplots(row_num, col_num, figsize=(16, 7))
    for i in range(len(df)):
        day_df = df.iloc[i]
        bins = day_df['hist_bins']
        cnts = day_df['hist_cnts']
        cnts = cnts / cnts.sum()

        ax = axs[int(i / col_num), i % col_num]
        ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), facecolor='red')

    plot_tail(f'Factor Distribution' , factor_name , benchmark , show , suptitle = True)
    return fig

@multi_factor_plot
def plot_factor_qtile(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df.columns.rename('quantile_name', inplace=True)
    df = df.set_index('date').stack().rename('quantile_value').reset_index(drop=False) # type: ignore
    # df = df.assign(date=pd.to_datetime(df['date'].astype(str)), format='%Y%m%d')

    ax = sns.lineplot(x='date', y='quantile_value', hue='quantile_name', data=df)
    plt.grid()
    # ax.set_title(f'Factor Quantile for [{factor_name}]')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    plt.grid()
    plot_tail(f'Factor Quantile' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_top_grp_perf_year(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    group = df['group'].values[0]
    df = df.drop(columns=['group' , 'abs_avg'])
    pct_cols = ['sum' , 'avg' , 'year_ret', 'std', 'cum_mdd']
    flt_cols = ['ir']
    df = df.assign(**df.loc[:,pct_cols].map(lambda x:f'{x:.3%}') , **df.loc[:,flt_cols].map(lambda x:f'{x:.ff}'))
    df.columns = [col.capitalize() for col in df.columns]
    plot_table(df.set_index('Year'))
    '''
    df.loc[pct_cols] = df.loc[pct_cols].map(lambda x: f'{x:.3%}' if not pd.isnull(x) else None)
    df.loc[['ir']] = df.loc[['ir']].map(lambda x: f'{x:.3f}' if not pd.isnull(x) else None)
    ax = plt.table(cellText=df.values,
                   colLabels=df.columns.tolist(), rowLabels=df.index.tolist(),
                   rowLoc='center', loc='center')
    ax.scale(1, 2)
    plt.axis('off')
    '''

    plot_tail(f'Annualized top Group ({group}) Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_ic_year(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    direction  = 'negative (-1)' if df['direction'].values[0] < 0 else 'positive (+1)'
    df['direction'] = df['direction'].astype(str)
    df['direction'] = direction
    df = df.rename(columns={'avg': 'IC_avg', 'std': 'IC_std','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, 
                   errors='raise').drop(columns=['sum' , 'year_ret']).rename(columns={'year':'Year'})
    flt_cols = ['IC_avg' , 'IC_std' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg']
    df = df.assign(**df.loc[:,flt_cols].map(lambda x:f'{x:.ff}'))
    plot_table(df.set_index('Year'))

    '''
    flt_cols = ['IC_avg' , 'IC_std' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg']
    df.loc[flt_cols] = df.loc[flt_cols].map(lambda x: f'{x:.3f}' if not pd.isnull(x) else None)
    ax = plt.table(cellText=df.values,
                   colLabels=df.columns.tolist(), rowLabels=df.index.to_list(),
                   rowLoc='center', loc='center')
    ax.scale(1, 2)
    plt.axis('off')
    '''
    
    plot_tail('Year IC' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_ic_curve(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    
    df = df.set_index('date')
    fig, ax1 = plt.subplots()  

    for col in df.columns.tolist():
        if col.startswith('ma_'): ax1.plot(df.index, df[col], label=col)  
    ax1.bar(df.index, df['ic'], color='b', label='IC')
    
    ax1.set_ylabel('period IC', color='b')  
    ax1.tick_params('y', colors='b')  
    ax1.legend(loc='upper left')  
    plt.xticks(rotation=45)  
    
    ax2 = ax1.twinx()  
    ax2.plot(df.index, df['cum_ic'], 'r-', label='Cum IC (right)')  
    
    ax2.set_ylabel('Cummulative IC', color='r')  
    ax2.tick_params('y', colors='r')  
    ax2.legend(loc='upper right')  

    ax1.grid()
    
    plot_tail('IC Curve' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_industry_ic(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    
    df = df.rename(columns={'avg':'IC_avg','ir':'ICIR'})
    df.sort_values(['ICIR'], ascending=False, inplace=True)

    fig, ax1 = plt.subplots(figsize=(16, 7))
    
    plt.bar(df['industry'], df['ICIR'], color='burlywood', alpha=0.5)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(['ICIR'], loc='upper left')
    plt.xticks(rotation=45)  

    ax2 = ax1.twinx()
    plt.plot(df['industry'], df['IC_avg'], color='darkred')
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    plt.subplots_adjust(bottom=0.3)
    ax2.legend(['Avg IC'], loc='upper right')

    ax1.grid()

    plot_tail('Industry IC & ICIR' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_pnl(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    df = df.pivot_table(index='date' , columns='weight_type' , values='cum_ret')
    df.index = df.index.astype(str)
    weight_type_list = df.columns.tolist()
    for weight_type in weight_type_list:
        plt.plot(df.index , df[weight_type], label=weight_type)

    plt.legend()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.2%}'))
    plt.grid()

    plot_tail('Cummulative Long-Short PnL' , factor_name , benchmark , show , suptitle = False)
    return fig