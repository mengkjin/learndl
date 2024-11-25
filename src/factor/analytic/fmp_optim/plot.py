import numpy as np
import pandas as pd

from plottable import ColumnDefinition
from typing import Any , Optional

from src.factor.util.plot import plot_table , get_twin_axes , set_xaxis , set_yaxis, PlotMultipleData , PlotFactorData
    
DROP_KEYS  = ['prefix' , 'factor_name' , 'benchmark' , 'strategy' , 'suffix']
MAJOR_KEYS = ['prefix' , 'factor_name' , 'benchmark' , 'strategy']

def plot_optim_frontface(data : pd.DataFrame , show = False):
    num_per_page : int | Any = 32 // data.groupby('factor_name').size().max()
    if num_per_page == 0: num_per_page = 1
    num_groups : int | Any = data.groupby('factor_name').ngroups
    num_pages  : int | Any = num_groups // num_per_page + (1 if num_groups % num_per_page > 0 else 0)
    group_plot = PlotMultipleData(data , group_key = 'factor_name' , max_num = num_per_page)
    for i , sub_data in enumerate(group_plot):     
        full_title = f'Optim FMP Front Face for Factors (P{i+1}/{num_pages})'
        with PlotFactorData(sub_data , drop = [] , name_key = None , show = show , full_title = full_title) as (df , fig):
            df = df.reset_index([i for i in df.index.names if i])
            df['strategy'] = df.apply(lambda x:'.'.join(x[col] for col in MAJOR_KEYS) , axis=1)
            plot_table(df.set_index('strategy') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] ,
                fontsize = 7 , index_width = 3 ,
                column_definitions = [ColumnDefinition(name='Lag'       , width=0.5) ,
                                      ColumnDefinition(name='Mdd_period', width=2)] , 
                stripe_by = ['factor_name' , 'benchmark'] ,
                ignore_cols = ['prefix' , 'factor_name' , 'benchmark' , 'suffix'])
    return group_plot.fig_dict

def plot_optim_perf_curve(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Accumulative Performance') as (df , fig):
            ax1 , ax2 = get_twin_axes(fig , 111)

            ax1.plot(df.index, df['pf'], label='portfolio')  
            ax1.plot(df.index, df['bm'], label='benchmark')  
            ax1.legend(loc='upper left')  
        
            ax2.plot(df.index, df['excess'], 'r-', )
            ax2.fill_between(df.index, df['excess'] , color='r', alpha=0.5 , label='Cum Excess (right)')
            ax2.legend(loc='upper right')  

            set_xaxis(ax2 , df.index , title = 'Trade Date')  
            set_yaxis(ax1 , format='pct' , digits=2 , title='Cummulative Return' , title_color='b' , tick_color='b')
            set_yaxis(ax2 , format='pct' , digits=2 , title='Cummulative Excess Return' , title_color='r' , tick_color='r' , tick_pos=None)
    return group_plot.fig_dict

def plot_optim_perf_drawdown(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Excess Drawdown') as (df , fig):
            ax1 , ax2 = get_twin_axes(fig , 111)

            ax1.plot(df.index, df['excess'], label='excess')  
            ax1.legend(loc='upper left')  
            
            ax2.plot(df.index, df['drawdown'], 'g', )
            ax2.fill_between(df.index, df['drawdown'] , color='g', alpha=0.5 , label='Drawdown (right)')
            ax2.legend(loc='upper right')  
            
            set_xaxis(ax1 , df.index , title = 'Trade Date')
            set_yaxis(ax1 , format='pct' , digits=2 , title='Cummulative Excess' , title_color='b' , tick_color='b')
            set_yaxis(ax2 , format='pct' , digits=2 , title='Drawdown' , title_color='g' , tick_color='g' , tick_pos=None)
    return group_plot.fig_dict

def plot_optim_perf_lag(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS , show = show and i == 0, 
                            title = 'Optim FMP Cumulative Lag Performance') as (df , fig):
            ax1 , ax2 = get_twin_axes(fig , 111)
            assert all([col.startswith('lag') for col in df.columns]) , df.columns
            [ax1.plot(df.index , df[col], label=col) for col in df.columns if col != 'lag_cost']
            ax1.legend(loc = 'upper left')

            

            ax2.plot(df.index, df['lag_cost'], 'r-', )
            ax2.fill_between(df.index, df['lag_cost'] , color='r', alpha=0.5 , label='Lag Cost (right)')
            ax2.legend(loc='upper right')  

            set_xaxis(ax1 , df.index , title = 'Trade Date')
            set_yaxis(ax1 , format='pct' , digits=2 , title='Cummulative Return' , title_color='b' , tick_color='b')
            set_yaxis(ax2 , format='pct' , digits=2 , title='Cummulative Lag Cost' , title_color='r' , tick_color='r' , 
                      tick_pos=None)

    return group_plot.fig_dict

def plot_optim_perf_year(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Year Performance') as (df , fig):
            plot_table(df.set_index('year') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] , 
                column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                stripe_by = 1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_optim_perf_month(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = [] , show = show and i == 0,  
                            title = 'FMP Month Performance') as (df , fig):
            plot_table(df.set_index('month') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] , 
                column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                stripe_by = 1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_optim_exp_style(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Style Exposure' , suptitle=True) as (df , fig):
            lay_out = (2, 5)
            assert len(df.columns) <= lay_out[0] * lay_out[1]
            for i , col in enumerate(df.columns): 
                ax = fig.add_subplot(*lay_out , i + 1 , frameon = False)
                ax.plot(df.index , df[col], label=col)
                ax.fill_between(df.index, df[col] , color='b', alpha=0.5)
                set_yaxis(ax , format='flt' , digits=2 , tick_size= 8  , tick_length=0 , tick_pos = 'left')
                set_xaxis(ax , df.index , tick_size= 8 , tick_length=0 , grid=False)
                ax.set_title(col.capitalize())
                ax.tick_params(left=False, right=False, top=False, bottom=False)

            fig.autofmt_xdate(rotation = 45)
    return group_plot.fig_dict

def plot_optim_exp_indus(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Industry Exposure' , suptitle=True) as (df , fig):
            lay_out = (5, 7)
            assert len(df.columns) <= lay_out[0] * lay_out[1]
            for i , col in enumerate(df.columns): 
                ax = fig.add_subplot(*lay_out , i + 1 , frameon = False)
                ax.plot(df.index , df[col], label=col)
                ax.fill_between(df.index, df[col] , color='b', alpha=0.5)
                set_yaxis(ax , format='pct' , digits=2 , tick_size= 8  , tick_length=0 , tick_pos = 'left')
                set_xaxis(ax , df.index , tick_size= 8 , tick_length=0 , grid=False)
                ax.set_title(col.capitalize())
                ax.tick_params(left=False, right=False, top=False, bottom=False)

            fig.autofmt_xdate(rotation = 45)
    return group_plot.fig_dict

def plot_optim_attrib_source(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Cumulative Attribution') as (df , fig):
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[col], label=col) for col in df.columns]
            ax.legend(loc = 'upper left')
            set_xaxis(ax , df.index , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title='Cumulative Attribution' , title_color='b' , tick_color='b')
            
    return group_plot.fig_dict

def plot_optim_attrib_style(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = DROP_KEYS ,  show = show and i == 0, 
                            title = 'Optim FMP Style Attribution' , suptitle=True) as (df , fig):
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[col], label=col) for col in df.columns]
            ax.legend(loc = 'upper left')

            set_xaxis(ax , df.index , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title='Cumulative Attribution' , title_color='b' , tick_color='b')
            
    return group_plot.fig_dict