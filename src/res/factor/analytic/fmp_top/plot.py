import pandas as pd

import src.func.plot as plot

from plottable import ColumnDefinition
from typing import Any

DROP_KEYS  = ['prefix' , 'factor_name' , 'benchmark' , 'strategy' , 'suffix']
MAJOR_KEYS = ['prefix' , 'factor_name' , 'benchmark' , 'strategy' , 'suffix']

def _strategy_name(keys : list[str]):
    def wrapper(x) -> str:
        return '.'.join(x[col] for col in keys)
    return wrapper

def df_strategy(df : pd.DataFrame) -> pd.Series:
    keys = [i for i in MAJOR_KEYS if i in df.columns or i in df.index.names]
    names = df.apply(_strategy_name(keys) , axis=1)
    assert isinstance(names , pd.Series) , f'names must be a pandas series, but got {type(names)}'
    return names

def plot_top_frontface(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    num_per_page : int | Any = 32 // data.groupby('factor_name').size().max()
    if num_per_page == 0: 
        num_per_page = 1
    num_groups : int | Any = data.groupby('factor_name').ngroups
    num_pages  : int | Any = num_groups // num_per_page + (1 if num_groups % num_per_page > 0 else 0)
    group_plot = plot.PlotMultipleData(data , group_key = 'factor_name' , max_num = num_per_page)
    for i , sub_data in enumerate(group_plot):     
        full_title = f'{title_prefix} Front Face (P{i+1}/{num_pages})'
        with plot.PlotFactorData(sub_data , drop = [] , name_key = None , show = show , full_title = full_title) as (df , fig):
            df = df.reset_index([i for i in df.index.names if i])
            df['strategy'] = df_strategy(df)
            plot.plot_table(df.set_index('strategy') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] ,
                            fontsize = 7 , index_width = 4 ,
                            column_definitions = [ColumnDefinition(name='lag'       , width=0.5) ,
                                                ColumnDefinition(name='mdd_period', width=1.4)] , 
                            stripe_by = ['factor_name' , 'benchmark'] , 
                            ignore_cols = ['prefix' , 'factor_name' , 'benchmark' , 'suffix'])
    return group_plot.fig_dict

def plot_top_perf_curve(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , show = show and i == 0, 
                                 title = f'{title_prefix} Accumulative Return') as (df , fig):
    
            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = plot.sns_lineplot(df , x='trade_date' , y='pf' , hue='topN')

            plot.set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b')
            
    return group_plot.fig_dict

def plot_top_perf_excess(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , show = show and i == 0, title = f'{title_prefix} Accumulative Excess Return') as (df , fig):
    
            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = plot.sns_lineplot(df , x='trade_date' , y='excess' , hue='topN')

            plot.set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Excess Return' , title_color='b')

    return group_plot.fig_dict

def plot_top_perf_drawdown(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , show = show and i == 0, 
                                 title = f'{title_prefix} Performance Drawdown') as (df , fig):

            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')

            if df['topN'].nunique() > 1:
                ax1 = plot.sns_lineplot(df , x='trade_date' , y='drawdown' , hue='topN')
                plot.set_yaxis(ax1 , format='pct' , digits=2 , title = 'Cummulative Drawdown' , title_color='b')
                plot.set_xaxis(ax1 , df['trade_date'].unique() , title = 'Trade Date')
            else:
                ax1 , ax2 = plot.get_twin_axes(fig , 111)
                
                ax1.plot(df['trade_date'], df['drawdown'], 'grey', label='Drawdown (left)')  
                ax1.legend(loc='upper left')

                for col in df.columns:
                    if col not in ['drawdown' , 'trade_date' , 'topN']:
                        ax2.plot(df['trade_date'] , df[col] , label=col)
                ax2.legend(loc='upper right')  

                plot.set_xaxis(ax1 , df['trade_date'] , title = 'Trade Date')
                plot.set_yaxis(ax1 , format='pct' , digits=2 , title = 'Drawdown' , title_color='b')
                plot.set_yaxis(ax2 , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b' , tick_pos=None)
                
    return group_plot.fig_dict

def plot_top_perf_excess_drawdown(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , show = show and i == 0, title = f'{title_prefix} Excess Drawdown') as (df , fig):

            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = plot.sns_lineplot(df , x='trade_date' , y='drawdown' , hue='topN')
            
            plot.set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Excess Drawdown' , title_color='b')
            
    return group_plot.fig_dict

def plot_top_perf_year(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data ,drop = [] , show = show and i == 0,  
                                 title = f'{title_prefix} Year Performance') as (df , fig):
            plot.plot_table(df.set_index('year') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] ,    
                column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] , 
                stripe_by = 1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_top_perf_month(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , drop = [] , show = show and i == 0,  
                            title = f'{title_prefix} Month Performance') as (df , fig):
            plot.plot_table(df.set_index('month') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] , 
                            column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                            stripe_by = 1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_top_exp_style(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , drop = [] ,show = show and i == 0, title = f'{title_prefix} Style Exposure') as (df , fig):
            df = df.groupby(['strategy'] , observed=True).mean().reset_index().\
                melt(id_vars=['strategy'] , var_name='style' , value_name='exposure')
            ax = plot.sns_barplot(df , x='style' , y='exposure' , hue='strategy')

            plot.set_xaxis(ax , title = 'Style')
            plot.set_yaxis(ax , format='flt' , digits=1 , title = 'Style Exposure' , title_color='b')

    return group_plot.fig_dict

def plot_top_exp_indus(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , drop = [] ,show = show and i == 0, title = f'{title_prefix} Industry Exposure') as (df , fig):
            df = df.groupby(['strategy'] , observed=True).mean().reset_index().\
                melt(id_vars=['strategy'] , var_name='industry' , value_name='deviation')
            df_mean : pd.Series | Any = df.groupby(['industry'] , observed=True)['deviation'].mean()
            df_mean = df_mean.rename('mdev')
            df = df.merge(df_mean , on='industry').sort_values(['mdev' , 'strategy'] , ascending=[False , True]).drop(columns=['mdev'])
            ax = plot.sns_barplot(df , x='industry' , y='deviation' , hue='strategy' , legend='upper right')

            plot.set_xaxis(ax , title = 'Industry')
            plot.set_yaxis(ax , format='pct' , digits=1 , title = 'Industry Deviation' , title_color='b')

    return group_plot.fig_dict

def plot_top_attrib_source(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , drop = [] , show = show and i == 0, title = f'{title_prefix} Cumulative Attribution') as (df , fig):
            df = df.reset_index().set_index('source')[['strategy' , 'contribution']].\
                loc[['tot' , 'excess' , 'market' , 'industry' , 'style' , 'specific' , 'cost']]
            ax = plot.sns_barplot(df , x='source' , y='contribution' , hue='strategy' , legend='upper right')

            plot.set_xaxis(ax , df.index , title = 'Source' , num_ticks=len(df.index))
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Total Attribution' , title_color='b')
            
    return group_plot.fig_dict

def plot_top_attrib_style(data : pd.DataFrame , show = False , title_prefix = 'Top Port'):
    group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with plot.PlotFactorData(sub_data , drop = [] , show = show and i == 0 , title = f'{title_prefix} Style Attribution') as (df , fig):
            df = df.reset_index().set_index('style').filter(items=['strategy' , 'contribution'])
            ax = plot.sns_barplot(df , x='style' , y='contribution' , hue='strategy' , legend='upper right')

            plot.set_xaxis(ax , df.index , title = 'Style' , num_ticks=len(df.index))
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Total Attribution' , title_color='b')

    return group_plot.fig_dict