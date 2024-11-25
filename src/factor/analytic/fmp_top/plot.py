import numpy as np
import pandas as pd

from plottable import ColumnDefinition
from typing import Any , Optional

from src.factor.util.plot import plot_table , set_xaxis , set_yaxis , PlotMultipleData , PlotFactorData , sns_lineplot , sns_barplot
    
DROP_KEYS  = ['prefix' , 'factor_name' , 'benchmark' , 'strategy' , 'suffix']
MAJOR_KEYS = ['prefix' , 'factor_name' , 'benchmark' , 'strategy']

def plot_top_frontface(data : pd.DataFrame , show = False):
    num_per_page : int | Any = 32 // data.groupby('factor_name').size().max()
    if num_per_page == 0: num_per_page = 1
    num_groups : int | Any = data.groupby('factor_name').ngroups
    num_pages  : int | Any = num_groups // num_per_page + (1 if num_groups % num_per_page > 0 else 0)
    group_plot = PlotMultipleData(data , group_key = 'factor_name' , max_num = num_per_page)
    for i , sub_data in enumerate(group_plot):     
        full_title = f'TopPort FMP Front Face for Factors (P{i+1}/{num_pages})'
        with PlotFactorData(sub_data , drop = [] , name_key = None , show = show , full_title = full_title) as (df , fig):
            df = df.reset_index([i for i in df.index.names if i])
            df['strategy'] = df.apply(lambda x:'.'.join(x[col] for col in MAJOR_KEYS) , axis=1)
            plot_table(df.set_index('strategy') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] ,
                fontsize = 7 , index_width = 3 ,
                column_definitions = [ColumnDefinition(name='lag'       , width=0.5) ,
                                      ColumnDefinition(name='mdd_period', width=2)] , 
                stripe_by = ['factor_name' , 'benchmark'] , 
                ignore_cols = ['prefix' , 'factor_name' , 'benchmark' , 'suffix'])
    return group_plot.fig_dict

def plot_top_perf_curve(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , show = show and i == 0, title = 'TopPort Accumulative Return') as (df , fig):
    
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = sns_lineplot(df , x='trade_date' , y='pf' , hue='topN')

            set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b')
            
    return group_plot.fig_dict

def plot_top_perf_excess(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , show = show and i == 0, title = 'TopPort Accumulative Excess Return') as (df , fig):
    
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = sns_lineplot(df , x='trade_date' , y='excess' , hue='topN')

            set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Excess Return' , title_color='b')

    return group_plot.fig_dict

def plot_top_perf_drawdown(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name','benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , show = show and i == 0, title = 'TopPort Excess Drawdown') as (df , fig):

            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = sns_lineplot(df , x='trade_date' , y='drawdown' , hue='topN')
            
            set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Excess Drawdown' , title_color='b')
            
    return group_plot.fig_dict

def plot_top_perf_year(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data ,drop = [] , show = show and i == 0,  
                            title = 'TopPort Year Performance') as (df , fig):
            plot_table(df.set_index('year') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] , 
                column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] , 
                stripe_by = 1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_top_perf_month(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = MAJOR_KEYS)
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = [] , show = show and i == 0,  
                            title = 'TopPort Month Performance') as (df , fig):
            plot_table(df.set_index('month') , 
                pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                flt_cols = ['ir','calmar'] , 
                column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                stripe_by = 1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_top_exp_style(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = [] ,show = show and i == 0, title = 'TopPort Style Exposure') as (df , fig):
            df = df.groupby(['strategy'] , observed=True).mean().reset_index().\
                melt(id_vars=['strategy'] , var_name='style' , value_name='exposure')
            ax = sns_barplot(df , x='style' , y='exposure' , hue='strategy')

            set_xaxis(ax , title = 'Style')
            set_yaxis(ax , format='flt' , digits=1 , title = 'Style Exposure' , title_color='b')

    return group_plot.fig_dict

def plot_top_exp_indus(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = [] ,show = show and i == 0, title = 'TopPort Industry Exposure') as (df , fig):
            df = df.groupby(['strategy'] , observed=True).mean().reset_index().\
                melt(id_vars=['strategy'] , var_name='industry' , value_name='deviation')
            df_mean = df.groupby(['industry'] , observed=True)['deviation'].mean().rename('mdev')
            df = df.merge(df_mean , on='industry').sort_values(['mdev' , 'strategy'] , ascending=[False , True]).drop(columns=['mdev'])
            ax = sns_barplot(df , x='industry' , y='deviation' , hue='strategy' , legend='upper right')

            set_xaxis(ax , title = 'Industry')
            set_yaxis(ax , format='pct' , digits=1 , title = 'Industry Deviation' , title_color='b')

    return group_plot.fig_dict

def plot_top_attrib_source(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = [] , show = show and i == 0, title = 'TopPort Cumulative Attribution') as (df , fig):
            df = df.reset_index().set_index('source')[['strategy' , 'contribution']].\
                loc[['tot' , 'excess' , 'market' , 'industry' , 'style' , 'specific' , 'cost']]
            ax = sns_barplot(df , x='source' , y='contribution' , hue='strategy' , legend='upper right')

            set_xaxis(ax , df.index , title = 'Source' , num_ticks=len(df.index))
            set_yaxis(ax , format='pct' , digits=2 , title = 'Total Attribution' , title_color='b')
            
    return group_plot.fig_dict

def plot_top_attrib_style(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = [] , show = show and i == 0 , title = 'TopPort Style Attribution') as (df , fig):
            df = df.reset_index().set_index('style')[['strategy' , 'contribution']]
            ax = sns_barplot(df , x='style' , y='contribution' , hue='strategy' , legend='upper right')

            set_xaxis(ax , df.index , title = 'Style' , num_ticks=len(df.index))
            set_yaxis(ax , format='pct' , digits=2 , title = 'Total Attribution' , title_color='b')

    return group_plot.fig_dict