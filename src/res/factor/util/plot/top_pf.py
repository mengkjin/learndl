import pandas as pd

from plottable import ColumnDefinition
from typing import Any

import src.func.plot as plot
from .plot_basic import PlotDfFigIterator

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

class Plotter:
    def __init__(self , title_prefix = 'Top Port'):
        self.plot_iter = PlotDfFigIterator(title_prefix)

    def plot_frontface(self , data : pd.DataFrame , show = False , title_prefix = None):
        num_per_page : int | Any = max(32 // data.groupby('factor_name').size().max() , 1)
        num_groups : int | Any = data.groupby('factor_name').ngroups
        num_pages  : int | Any = num_groups // num_per_page + (1 if num_groups % num_per_page > 0 else 0)
        self.plot_iter.set_args(data , show , title_prefix , 'Front Face' , ['factor_name'] , drop_keys = False , drop_cols = [] , num_groups_per_iter = num_per_page , num_pages = num_pages)

        for df , fig in self.plot_iter.iter():     
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
        return self.plot_iter.figs

    def plot_perf_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Accumulative Return' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = plot.sns_lineplot(df , x='trade_date' , y='pf' , hue='topN')

            plot.set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b')
                
        return self.plot_iter.figs

    def plot_perf_excess(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Accumulative Excess Return' , ['factor_name' , 'benchmark'])

        for df , fig in self.plot_iter.iter():
            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = plot.sns_lineplot(df , x='trade_date' , y='excess' , hue='topN')

            plot.set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Excess Return' , title_color='b')

        return self.plot_iter.figs

    def plot_perf_drawdown(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Performance Drawdown' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
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
                    
        return self.plot_iter.figs

    def plot_perf_excess_drawdown(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Excess Drawdown' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            if 'topN' not in df.columns and 'topN' not in df.index.names: 
                df['topN'] = '~'
            df = df.reset_index().sort_values(['topN' , 'trade_date'])
            df['topN'] = 'Top' + df['topN'].astype(str).str.rjust(3 , fillchar=' ')
            ax = plot.sns_lineplot(df , x='trade_date' , y='drawdown' , hue='topN')
            plot.set_xaxis(ax , df['trade_date'].unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Excess Drawdown' , title_color='b')
        return self.plot_iter.figs

    def plot_perf_year(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Year Performance' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            plot.plot_table(df.set_index('year') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] ,    
                            column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] , 
                            stripe_by = 1 , emph_last_row=True)
        return self.plot_iter.figs

    def plot_perf_month(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Month Performance' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            plot.plot_table(df.set_index('month') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] , 
                            column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                            stripe_by = 1 , emph_last_row=True)
        return self.plot_iter.figs

    def plot_exp_style(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Style Exposure' , ['factor_name' , 'benchmark'] , drop_keys = False)
        for df , fig in self.plot_iter.iter():
            df = df.groupby(['strategy'] , observed=True).mean().reset_index().\
                melt(id_vars=['strategy'] , var_name='style' , value_name='exposure')
            ax = plot.sns_barplot(df , x='style' , y='exposure' , hue='strategy')

            plot.set_xaxis(ax , title = 'Style')
            plot.set_yaxis(ax , format='flt' , digits=1 , title = 'Style Exposure' , title_color='b')
        return self.plot_iter.figs

    def plot_exp_indus(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Industry Exposure' , ['factor_name' , 'benchmark'] , drop_keys = False)
        for df , fig in self.plot_iter.iter():
            df = df.groupby(['strategy'] , observed=True).mean().reset_index().\
                melt(id_vars=['strategy'] , var_name='industry' , value_name='deviation')
            df_mean : pd.Series | Any = df.groupby(['industry'] , observed=True)['deviation'].mean()
            df_mean = df_mean.rename('mdev')
            df = df.merge(df_mean , on='industry').sort_values(['mdev' , 'strategy'] , ascending=[False , True]).drop(columns=['mdev'])
            ax = plot.sns_barplot(df , x='industry' , y='deviation' , hue='strategy' , legend='upper right')

            plot.set_xaxis(ax , title = 'Industry')
            plot.set_yaxis(ax , format='pct' , digits=1 , title = 'Industry Deviation' , title_color='b')
        return self.plot_iter.figs

    def plot_attrib_source(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Cumulative Attribution' , ['factor_name' , 'benchmark'] , drop_keys = False)
        for df , fig in self.plot_iter.iter():
            df = df.reset_index().set_index('source')[['strategy' , 'contribution']].\
                loc[['tot' , 'excess' , 'market' , 'industry' , 'style' , 'specific' , 'cost']]
            ax = plot.sns_barplot(df , x='source' , y='contribution' , hue='strategy' , legend='upper right')

            plot.set_xaxis(ax , df.index , title = 'Source' , num_ticks=len(df.index))
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Total Attribution' , title_color='b')
        return self.plot_iter.figs

    def plot_attrib_style(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Style Attribution' , ['factor_name' , 'benchmark'] , drop_keys = False)
        for df , fig in self.plot_iter.iter():
            df = df.reset_index().set_index('style').filter(items=['strategy' , 'contribution'])
            ax = plot.sns_barplot(df , x='style' , y='contribution' , hue='strategy' , legend='upper right')

            plot.set_xaxis(ax , df.index , title = 'Style' , num_ticks=len(df.index))
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Total Attribution' , title_color='b')
        return self.plot_iter.figs