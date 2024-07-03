import pandas as pd

from plottable import ColumnDefinition
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from typing import Any , Optional

from ..util.plot import multi_factor_plot , plot_head , plot_tail , plot_table , plot_xaxis
from ...basic import AVAIL_BENCHMARKS

@multi_factor_plot
def plot_fmp_perf_lag(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    ax = fig.add_subplot(111)
    df = df.set_index('trade_date')
    for col in df.columns: 
        if col != 'lag_cost': ax.plot(df.index , df[col], label=col)
    ax.legend(loc = 'upper left')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))
    plot_xaxis(ax , df.index)

    if 'lag_cost' in df.columns:
        ax2 : Axes | Any = ax.twinx()  
        ax2.plot(df.index, df['lag_cost'], 'r-', )
        ax2.fill_between(df.index, df['lag_cost'] , color='r', alpha=0.5 , label='Lag Cost (right)')
        
        ax2.set_ylabel('Cummulative Lag Cost', color='r')  
        ax2.tick_params('y', colors='r')  
        ax2.legend(loc='upper right')  
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))  

    plot_tail(f'FMP Cumulative Lag Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_perf_year(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    plot_table(df.set_index('year') , 
               pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
               flt_cols = ['ir','calmar'] , 
               column_definitions = [ColumnDefinition(name='Mdd_period', width=2)])
    plot_tail(f'FMP Year Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_perf_month(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    plot_table(df.set_index('month') , 
               pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
               flt_cols = ['ir','calmar'])
    plot_tail(f'FMP Month Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_perf_curve(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index('trade_date').rename(columns={'pf':'portfolio','bm':'benchmark'})
    ax1 = fig.add_subplot(111)

    for col in ['portfolio','benchmark']: 
        ax1.plot(df.index, df[col], label=col)  
    
    ax1.set_ylabel('Cummulative Return', color='b')  
    ax1.tick_params('y', colors='b')  
    ax1.legend(loc='upper left')  
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))  
    plot_xaxis(ax1 , df.index)

    ax2 : Axes | Any = ax1.twinx()  
    ax2.plot(df.index, df['excess'], 'r-', )
    ax2.fill_between(df.index, df['excess'] , color='r', alpha=0.5 , label='Cum Excess (right)')
    
    ax2.set_ylabel('Cummulative Excess Return', color='r')  
    ax2.tick_params('y', colors='r')  
    ax2.legend(loc='upper right')  
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))  

    plot_tail(f'FMP Accumulative Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_perf_drawdown(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index('trade_date')
    ax1 = fig.add_subplot(111)

    ax1.plot(df.index, df['excess'], label='excess')  
    ax1.set_ylabel('Cummulative Excess', color='b')  
    ax1.tick_params('y', colors='b')  
    ax1.legend(loc='upper left')  
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))  
    plot_xaxis(ax1 , df.index)

    ax2 : Axes | Any = ax1.twinx()  
    ax2.plot(df.index, df['drawdown'], 'g', )
    ax2.fill_between(df.index, df['drawdown'] , color='g', alpha=0.5 , label='Drawdown (right)')
    
    ax2.set_ylabel('Drawdown', color='r')  
    ax2.tick_params('y', colors='r')  
    ax2.legend(loc='upper right')  
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))  

    plot_tail(f'FMP Excess Drawdown' , factor_name , benchmark , show , suptitle = False)
    return fig


@multi_factor_plot
def plot_fmp_style_exp(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    df = df.set_index('trade_date')
    lay_out = (2, 5)
    assert len(df.columns) <= lay_out[0] * lay_out[1]
    for i , col in enumerate(df.columns): 
        ax = fig.add_subplot(*lay_out , i + 1 , frameon = False)
        ax.plot(df.index , df[col], label=col)
        ax.fill_between(df.index, df[col] , color='b', alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2f}'))
        ax.yaxis.set_tick_params(labelsize = 8 , length = 0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_tick_params(labelsize = 8 , length = 0)
        ax.set_title(col.capitalize())
        ax.set_xticks(df.index[::max(1,len(df.index)//10)])

    fig.autofmt_xdate(rotation = 45)
    plot_tail(f'FMP Style Exposure' , factor_name , benchmark , show , suptitle = True)
    return fig

@multi_factor_plot
def plot_fmp_industry_exp(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    df = df.set_index('trade_date')
    lay_out = (5, 7)
    assert len(df.columns) <= lay_out[0] * lay_out[1]
    for i , col in enumerate(df.columns): 
        ax = fig.add_subplot(*lay_out , i + 1 , frameon = False)
        ax.plot(df.index , df[col], label=col)
        ax.fill_between(df.index, df[col] , color='b', alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))
        ax.yaxis.set_tick_params(labelsize = 6 , length = 0)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_tick_params(labelsize = 6 , length = 0)
        ax.set_title(col.capitalize() , fontsize = 8)
        ax.set_xticks(df.index[::max(1,len(df.index)//10)])
    fig.autofmt_xdate(rotation = 45)
    plot_tail(f'FMP Industry Deviation' , factor_name , benchmark , show , suptitle = True)
    return fig

@multi_factor_plot
def plot_fmp_attrib_source(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    ax = fig.add_subplot(111)
    df = df.set_index('trade_date').round(6)
    for col in ['tot' , 'market' , 'industry' , 'style' , 'specific' , 'cost']: 
        ax.plot(df.index , df[col], label=col)
    ax.legend(loc = 'upper left')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))   
    plot_xaxis(ax , df.index)
    plot_tail(f'FMP Cumulative Attribution' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_attrib_style(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    ax = fig.add_subplot(111)
    df = df.set_index('trade_date').round(6)

    [ax.plot(df.index , df[col], label=col) for col in df.columns]
    
    ax.legend(loc = 'upper left')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,y:f'{x:.2%}'))
    plot_xaxis(ax , df.index)

    plot_tail(f'FMP Cumulative Attribution' , factor_name , benchmark , show , suptitle = False)
    return fig

def plot_fmp_prefix(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , None , None)

    df = df.rename(columns={'factor_name':'factor'}).set_index('factor')
    # df['benchmark'] = pd.Categorical(df['benchmark'], categories = ['default'] + AVAIL_BENCHMARKS, ordered=True)  
    df = df.sort_values(['factor','benchmark'])  

    plot_table(df , 
               pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
               flt_cols = ['ir','calmar'] ,
               column_definitions = [ColumnDefinition(name='Factor'    , width=2) ,
                                     ColumnDefinition(name='Mdd_period', width=2)] , 
               emph_last_row=False , stripe_rows=df.groupby('factor')['benchmark'].count().to_list())
    plot_tail('FMP Prefix Information' , 'All Factors' , show = show , suptitle=False)
    return fig