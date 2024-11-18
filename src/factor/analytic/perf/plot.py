import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from typing import Any , Literal , Optional

from ...util.plot import CURRENT_SEABORN_VERSION , multi_factor_plot , plot_head , plot_tail , plot_table , plot_xaxis

@multi_factor_plot
def plot_decay_ic(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    df = df.set_index('lag_type')
    ax = fig.add_subplot(111)
    bars = ax.bar(df.index  , df['ic_mean'], label=factor_name)  
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2 , height ,
                 f'{height:.4f}' , ha = 'center' , va = 'top' if height < 0 else 'bottom')
    plot_xaxis(ax , df.index)
    plot_tail(f'Factor Average IC Decay' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_decay_grp_perf(df : pd.DataFrame , factor_name : Optional[str] = None , 
                        benchmark : Optional[str] = None , stat_type : Literal['ret' , 'ir'] = 'ret', show = False):
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
    plot_xaxis(ax , None)
    plot_tail(f'Factor Groups {stat_type.upper()} Decay' , factor_name , benchmark , show , suptitle = False)
    return fig

def plot_decay_grp_perf_ir(df : pd.DataFrame , factor_name : Optional[str] = None , 
                           benchmark : Optional[str] = None , show = False):
    return plot_decay_grp_perf(df , factor_name , benchmark , stat_type = 'ir' , show = show)

def plot_decay_grp_perf_ret(df : pd.DataFrame , factor_name : Optional[str] = None , 
                           benchmark : Optional[str] = None , show = False):
    return plot_decay_grp_perf(df , factor_name , benchmark , stat_type = 'ret' , show = show)
    
@multi_factor_plot
def plot_grp_perf(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df_0 = df.groupby(['group'] , observed=False)['date'].min().reset_index()
    df_0 = df_0.assign(date = df_0['date'].astype(str) , group_ret = 0.)

    df = df.assign(date = df['end'].astype(str)).loc[:,['date','group','group_ret']]
    df = pd.concat([df_0 , df]).set_index(['date', 'group']).\
        groupby('group' , observed=False)['group_ret'].\
        cumsum().rename('cum_ret').reset_index().set_index('date')

    if CURRENT_SEABORN_VERSION:
        ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=df,
                          palette=sns.diverging_palette(140, 10, sep=10, n=df['group'].nunique()))
    else:
        ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.2%}'))
    plot_xaxis(ax , df.index)
    plot_tail(f'Factor Group CumReturn' , factor_name , benchmark , show , suptitle = False)
    return fig
    
@multi_factor_plot
def plot_style_corr_box(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index(['date'])
    df.columns.rename('style_factor', inplace=True)
    df = df.stack().rename('factor_corr').reset_index(drop=False) # type: ignore

    ax = sns.boxplot(x='style_factor', y='factor_corr', data=df, width=0.3)
    plot_xaxis(ax , None)
    plot_tail(f'Factor Corr with Risk Styles' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_style_corr(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index(['date'])
    ax = fig.add_subplot(111)
    for style in df.columns.tolist(): ax.plot(df.index , df[style], label=style)
    plot_xaxis(ax , df.index)
    plot_tail(f'Factor Corr Curve with Risk Styles' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_distribution(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    assert not df['date'].duplicated().any()
    
    lay_out = (3, min(int(np.ceil(len(df) / 3)) , 3))
    for i in range(min(len(df) , lay_out[0] * lay_out[1])):
        ax = fig.add_subplot(*lay_out , i + 1)
        day_df = df.iloc[i]
        bins = day_df['hist_bins']
        cnts = day_df['hist_cnts'] / day_df['hist_cnts'].sum()
        ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), color='b' , alpha = 0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.0%}'))
        ax.yaxis.set_tick_params(labelsize = 8 , length = 0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.1f}'))
        ax.xaxis.set_tick_params(labelsize = 8 , length = 0)
        ax.set_title(str(day_df['date']))
    # fig.subplots_adjust(hspace=0.3)
    plot_tail(f'Factor Cross-Sectional Distribution' , factor_name , benchmark , show , suptitle = True)
    return fig

@multi_factor_plot
def plot_factor_qtile(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df.columns.rename('quantile_name', inplace=True)
    df = df.set_index('date').stack().rename('quantile_value').reset_index().set_index('date') # type: ignore
    # df = df.assign(date=pd.to_datetime(df['date'].astype(str)), format='%Y%m%d')

    ax = sns.lineplot(x='date', y='quantile_value', hue='quantile_name', data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
    plot_xaxis(ax , df.index)
    plot_tail(f'Factor Cross-Sectional Quantile' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_top_grp_perf_year(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    group = df['group'].unique()[0]
    df = df.drop(columns=['group' , 'abs_avg'])

    plot_table(df.set_index('year') , 
               pct_cols = ['sum' , 'avg' , 'year_ret', 'std', 'cum_mdd'] , 
               flt_cols = ['ir'] , pct_ndigit = 3)

    plot_tail(f'Factor Annualized Top Group ({group}) Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_ic_year(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    direction  = 'N(-)' if df['direction'].values[-1] < 0 else 'P(+)'
    df['direction'] = df['direction'].astype(str)
    df['direction'] = direction
    df = df.rename(columns={'avg': 'IC_avg', 'std': 'IC_std','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, 
                   errors='raise').drop(columns=['sum' , 'year_ret']).rename(columns={'year':'Year'})

    plot_table(df.set_index('Year') , flt_cols = ['IC_avg' , 'IC_std' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] ,
               capitalize=False)
    
    plot_tail('Factor Year IC' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_ic_curve(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    
    df = df.set_index('date')
    ax1 = fig.add_subplot(111)

    ax1.bar(df.index, df['ic'], color='b', label='IC')
    ax1.set_ylabel('period IC', color='b')  
    ax1.tick_params('y', colors='b') 
    colors = ['orange','purple','olive','pink','cyan','grey','azure','yellow']
    for col in df.columns.tolist():
        if col.startswith('ma_'): ax1.plot(df.index, df[col], color=colors.pop(0) , label=col)  
    ax1.legend(loc='upper left')  
    plot_xaxis(ax1 , df.index)

    ax2 : Axes | Any = ax1.twinx()  
    ax2.plot(df.index, df['cum_ic'], 'r-', label='Cum IC (right)')  
    ax2.set_ylabel('Cummulative IC', color='r')  
    ax2.tick_params('y', colors='r')  
    ax2.legend(loc='upper right')  
    
    plot_tail('Factor IC Curve' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_industry_ic(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    
    df = df.rename(columns={'avg':'IC_avg','ir':'ICIR'})
    df.sort_values(['IC_avg'], ascending=False, inplace=True)

    ax1 = fig.add_subplot(111)
    
    ax1.bar(df['industry'], df['IC_avg'], color='b', alpha=0.5)
    ax1.set_ylabel('Average IC', color='b')  
    ax1.tick_params('y', colors='b') 
    ax1.xaxis.set_ticks_position('bottom')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(['Avg IC'], loc='upper left')
    plot_xaxis(ax1 , None)

    ax2 : Axes | Any = ax1.twinx()
    ax2.plot(df['industry'], df['ICIR'], 'r-')
    ax2.set_ylabel('Average ICIR', color='r')  
    ax2.tick_params('y', colors='r')  
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.legend(['Avg ICIR'], loc='upper right')

    plot_tail('Factor Industry IC & IR' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_ic_monotony(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    df = df.pivot_table('stats_value' , 'group' , 'stats_name',observed=False).\
        rename(columns={'grp_ret':'RET','grp_ir':'IR'}).reset_index()
    ax1 = fig.add_subplot(111)
    ax1.bar(df['group'], df['RET'], color='b', alpha=0.5)
    ax1.set_ylabel('Average Ret', color='b')  
    ax1.tick_params('y', colors='b') 
    
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(['Avg Ret'], loc='upper left')
    ax1.grid()

    ax2 : Axes | Any = ax1.twinx()
    ax2.plot(df['group'], df['IR'], 'r-')
    ax2.set_ylabel('Grouped IR', color='r')  
    ax2.tick_params('y', colors='r')  
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.legend(['Avg IR'], loc='upper right')
    
    ax1.set_xticks([])
    ax2.set_xticks([])

    plot_tail('Factor Percentile Ret & IR' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_pnl(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    df = df.pivot_table(index='date' , columns='weight_type' , values='cum_ret')
    df.index = df.index.astype(str)
    weight_type_list = df.columns.tolist()
    ax = fig.add_subplot(111)
    for weight_type in weight_type_list:
        ax.plot(df.index , df[weight_type], label=weight_type)

    ax.legend()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.2%}'))
    plot_xaxis(ax , df.index)

    plot_tail('Factor Cummulative Long-Short PnL' , factor_name , benchmark , show , suptitle = False)
    return fig