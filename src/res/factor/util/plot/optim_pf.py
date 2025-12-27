import pandas as pd

from plottable import ColumnDefinition
from typing import Any

from src.proj.util import Plot
from .plot_basic import PlotDfFigIterator

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

class Plotter:
    def __init__(self , title_prefix = 'Optim Port'):
        self.plot_iter = PlotDfFigIterator(title_prefix)

    def plot_frontface(self , data : pd.DataFrame , show = False , title_prefix = None):
        num_per_page : int | Any = max(32 // data.groupby('factor_name').size().max() , 1)
        num_groups : int | Any = data.groupby('factor_name').ngroups
        num_pages  : int | Any = num_groups // num_per_page + (1 if num_groups % num_per_page > 0 else 0)
        self.plot_iter.set_args(data , show , title_prefix , 'Front Face' , ['factor_name'] , drop_keys = False , drop_cols = [] , num_groups_per_iter = num_per_page , num_pages = num_pages)
        for df , fig in self.plot_iter.iter():
            df = df.reset_index([i for i in df.index.names if i])
            df['strategy'] = df_strategy(df)
            Plot.plot_table(df.set_index('strategy') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] ,
                            fontsize = 7 , index_width = 4 ,
                            column_definitions = [ColumnDefinition(name='Lag'       , width=0.5) ,
                                ColumnDefinition(name='Mdd_period', width=1.5)] , 
                            stripe_by = ['factor_name' , 'benchmark'] ,
                            ignore_cols = ['prefix' , 'factor_name' , 'benchmark' , 'suffix'])
        return self.plot_iter.figs

    def plot_perf_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Accumulative Performance' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            ax1 , ax2 = Plot.get_twin_axes(fig , 111)
            ax1.plot(df.index, df['pf'], label='portfolio')  
            ax1.plot(df.index, df['bm'], label='benchmark')  
            ax1.legend(loc='upper left')  
            ax2.plot(df.index, df['excess'], 'r-', )
            ax2.fill_between(df.index, df['excess'] , color='r', alpha=0.5 , label='Cum Excess (right)')
            ax2.legend(loc='upper right')  

            Plot.set_xaxis(ax1 , df.index , title = 'Trade Date')
            Plot.set_yaxis(ax1 , format='pct' , digits=2 , title='Cummulative Return' , title_color='b' , tick_color='b')
            Plot.set_yaxis(ax2 , format='pct' , digits=2 , title='Cummulative Excess Return' , title_color='r' , tick_color='r' , tick_pos=None)
        return self.plot_iter.figs

    def plot_perf_drawdown(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Performance Drawdown' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            ax1 , ax2 = Plot.get_twin_axes(fig , 111)
            
            ax1.plot(df.index, df['drawdown'], 'grey', label='Drawdown (left)')  
            ax1.legend(loc='upper left')

            for col in df.columns:
                if col != 'drawdown':
                    ax2.plot(df.index , df[col] , label=col)
            ax2.legend(loc='upper right')  

            Plot.set_xaxis(ax1 , df.index , title = 'Trade Date')
            Plot.set_yaxis(ax1 , format='pct' , digits=2 , title = 'Cummulative Drawdown' , title_color='b')
            Plot.set_yaxis(ax2 , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b' , tick_pos=None)
                    
        return self.plot_iter.figs

    def plot_perf_excess_drawdown(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Excess Drawdown' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            ax1 , ax2 = Plot.get_twin_axes(fig , 111)

            ax1.plot(df.index, df['excess'], label='excess')  
            ax1.legend(loc='upper left')  
            
            ax2.plot(df.index, df['drawdown'], 'g', )
            ax2.fill_between(df.index, df['drawdown'] , color='g', alpha=0.5 , label='Drawdown (right)')
            ax2.legend(loc='upper right')  
            
            Plot.set_xaxis(ax1 , df.index , title = 'Trade Date')
            Plot.set_yaxis(ax1 , format='pct' , digits=2 , title='Cummulative Excess' , title_color='b' , tick_color='b')
            Plot.set_yaxis(ax2 , format='pct' , digits=2 , title='Drawdown' , title_color='g' , tick_color='g' , tick_pos=None)
        return self.plot_iter.figs

    def plot_perf_lag(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Cumulative Lag Performance' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            ax1 , ax2 = Plot.get_twin_axes(fig , 111)
            assert all([col.startswith('lag') for col in df.columns]) , df.columns
            [ax1.plot(df.index , df[col], label=col) for col in df.columns if col != 'lag_cost']
            ax1.legend(loc = 'upper left')

            ax2.plot(df.index, df['lag_cost'], 'r-', )

            ax2.fill_between(df.index, df['lag_cost'] , color='r', alpha=0.5 , label='Lag Cost (right)')
            ax2.legend(loc='upper right')  

            Plot.set_xaxis(ax1 , df.index , title = 'Trade Date')
            Plot.set_yaxis(ax1 , format='pct' , digits=2 , title='Cummulative Return' , title_color='b' , tick_color='b')
            Plot.set_yaxis(ax2 , format='pct' , digits=2 , title='Cummulative Lag Cost' , title_color='r' , tick_color='r' , 
                    tick_pos=None)

        return self.plot_iter.figs

    def plot_perf_year(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Year Performance' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            Plot.plot_table(df.set_index('year') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] , 
                            column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                            stripe_by = 1 , emph_last_row=True)
        return self.plot_iter.figs

    def plot_perf_month(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Month Performance' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            Plot.plot_table(df.set_index('month') , 
                            pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover'] , 
                            flt_cols = ['ir','calmar'] , 
                            column_definitions = [ColumnDefinition(name='Mdd_period', width=2)] ,
                            stripe_by = 1 , emph_last_row=True)
        return self.plot_iter.figs

    def plot_exp_style(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Style Exposure' , MAJOR_KEYS , suptitle=True)
        for df , fig in self.plot_iter.iter():
            lay_out = (2, 5)
            assert len(df.columns) <= lay_out[0] * lay_out[1] , (len(df.columns) , lay_out)
            for i , col in enumerate(df.columns): 
                ax = fig.add_subplot(*lay_out , i + 1 , frameon = False)
                ax.plot(df.index , df[col], label=col)
                ax.fill_between(df.index, df[col] , color='b', alpha=0.5)
                Plot.set_yaxis(ax , format='flt' , digits=2 , tick_size= 8  , tick_length=0 , tick_pos = 'left')
                Plot.set_xaxis(ax , df.index , tick_size= 8 , tick_length=0 , grid=False)
                ax.set_title(col.title())
                ax.tick_params(left=False, right=False, top=False, bottom=False)

            fig.autofmt_xdate(rotation = 45)
        return self.plot_iter.figs

    def plot_exp_indus(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Industry Exposure' , MAJOR_KEYS , suptitle=True)
        for df , fig in self.plot_iter.iter():
            lay_out = (5, 7)
            assert len(df.columns) <= lay_out[0] * lay_out[1] , (len(df.columns) , lay_out)
            for i , col in enumerate(df.columns): 
                ax = fig.add_subplot(*lay_out , i + 1 , frameon = False)
                ax.plot(df.index , df[col], label=col)
                ax.fill_between(df.index, df[col] , color='b', alpha=0.5)
                Plot.set_yaxis(ax , format='pct' , digits=2 , tick_size= 8  , tick_length=0 , tick_pos = 'left')
                Plot.set_xaxis(ax , df.index , tick_size= 8 , tick_length=0 , grid=False)
                ax.set_title(col.title())
                ax.tick_params(left=False, right=False, top=False, bottom=False)

            fig.autofmt_xdate(rotation = 45)
        return self.plot_iter.figs

    def plot_attrib_source(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Cumulative Attribution' , MAJOR_KEYS)
        for df , fig in self.plot_iter.iter():
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[col], label=col) for col in df.columns]
            ax.legend(loc = 'upper left')
            Plot.set_xaxis(ax , df.index , title = 'Trade Date')
            Plot.set_yaxis(ax , format='pct' , digits=2 , title='Cumulative Attribution' , title_color='b' , tick_color='b')
        return self.plot_iter.figs

    def plot_attrib_style(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Style Attribution' , MAJOR_KEYS , suptitle=True)
        for df , fig in self.plot_iter.iter():
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[col], label=col) for col in df.columns]
            ax.legend(loc = 'upper left')
            Plot.set_xaxis(ax , df.index , title = 'Trade Date')
            Plot.set_yaxis(ax , format='pct' , digits=2 , title='Cumulative Attribution' , title_color='b' , tick_color='b')
        return self.plot_iter.figs