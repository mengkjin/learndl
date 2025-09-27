import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from packaging import version
from plottable import Table , ColumnDefinition
from typing import Any , Literal

from . import display

CURRENT_SEABORN_VERSION = version.Version(getattr(sns , '__version__')) > version.Version('0.9.1')

sns.set_theme(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.family'] = ['monospace'] # ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei'] # for chinese
plt.rcParams['axes.unicode_minus'] = False

def new_figure(size = (16 , 7)):
    return plt.figure(figsize=size)

class PlotMultipleData:
    def __init__(self , data : pd.DataFrame , 
                 group_key : str | list[str],  max_num = 1 , **kwargs):
        self.data = data
        if isinstance(group_key , str): 
            group_key = [group_key]
        self.group_key = [i for i in group_key if i in data.columns or i in data.index.names]
        self.fig_dict : dict[str , Figure] = {}
        self.max_num   = max_num
    
    def __iter__(self):
        if self.max_num > 1:
            fig_i : int = 0
            stack_dfs : list[pd.DataFrame] = []
            for g , df in self.data.groupby(self.group_key , group_keys=True , observed=True):
                stack_dfs.append(df)
                if len(stack_dfs) >= self.max_num:
                    yield SubPlotData(self.group_key , pd.concat(stack_dfs) , self.fig_dict , f'P{fig_i+1}')
                    fig_i += 1
                    stack_dfs.clear()
            if stack_dfs:
                yield SubPlotData(self.group_key , pd.concat(stack_dfs) , self.fig_dict , f'P{fig_i+1}')
                stack_dfs.clear()
        else:
            for g , df in self.data.groupby(self.group_key , group_keys=True , observed=True):
                yield SubPlotData(self.group_key , df , self.fig_dict , self.group_name(g))

    def group_name(self , g):
        return '.'.join([str(s) for s in (g if isinstance(g , (list | tuple)) else [g])])

@dataclass  
class SubPlotData:
    group_by : str | list[str] | None
    sub_data : pd.DataFrame
    fig_dict : dict[str , Figure]
    fig_name : str 
    sub_key  : Any = None
    
    def __post_init__(self):
        if self.fig_name not in self.fig_dict:
            self.fig_dict[self.fig_name] = new_figure()

    def get_fig(self):
        return self.fig_dict[self.fig_name]

class PlotFactorData:
    def __init__(self , data : pd.DataFrame | SubPlotData , fig : Figure | None = None , 
                 name_key : list[str] | str | None = ['factor_name' , 'benchmark'] ,
                 drop : list[str] | str | None = ['prefix' , 'strategy' , 'factor_name' , 'benchmark' , 'suffix'] , 
                 title : str = '' , full_title : str = '' , show = False , 
                 sort_keys : list[str] = ['prefix' , 'factor_name' , 'benchmark' , 'strategy' , 'suffix' , 
                                          'trade_date' , 'model_date' , 'end' , 'start'] ,
                 dropna : bool | Literal['all' , 'any'] = True , rounding = 6 , suptitle = False):
        if isinstance(data , SubPlotData):
            data , name_key , fig  = data.sub_data , data.group_by , data.get_fig()

        self.raw_index = [i for i in data.index.names if i]
        self.raw_data  = data.reset_index(self.raw_index)
        self.drop_keys = drop if isinstance(drop , list) else [drop]
        self.sort_keys = [s for s in sort_keys if s in self.raw_data.columns]
        self.fig = fig if fig else new_figure()

        self.title = title
        self.full_title = full_title
        self.name_key = name_key
        self.show = show
        self.suptitle = suptitle
        self.dropna = dropna

        numeric_cols = self.raw_data.select_dtypes(include=['number']).columns
        self.raw_data[numeric_cols] = self.raw_data[numeric_cols].round(rounding)
        
        for col in self.raw_data.columns: 
            if col.endswith('date'): 
                self.raw_data[col] = self.raw_data[col].astype(str)

    def __enter__(self):
        df = self.raw_data.sort_values(self.sort_keys).drop(columns=self.drop_keys , errors='ignore')
        new_index = [i for i in self.raw_index if i not in self.drop_keys]
        df = df.set_index(new_index) if new_index else df.reset_index(drop = True)
        if self.dropna: 
            df = df.dropna(how='any' if self.dropna != 'all' else 'all')
        return df , self.fig

    def __exit__(self , exc_type , exc_value , traceback):
        if self.title or self.full_title:
            full_title = self.full_title if self.full_title else f'{self.title}{self.title_suffix()}'
            plt.suptitle(full_title , fontsize = 14) if self.suptitle else plt.title(full_title , fontsize = 14)
        plt.tight_layout()
        plt.close(self.fig)
        if self.show: 
            display.display(self.fig)

    def title_suffix(self):
        if self.name_key:
            name_key = self.name_key if isinstance(self.name_key , list) else [self.name_key]
            return f' for [{".".join([str(s) for s in self.raw_data[name_key].iloc[0].values])}]'
        else:
            return ''

def plot_table(df : pd.DataFrame , int_cols = None , pct_cols = None , flt_cols = None , capitalize = True , 
               index_width = 1. , fontsize = 8 , pct_ndigit = 2 , flt_ndigit = 3 , emph_last_row = False , 
               column_definitions : list[ColumnDefinition] | None = [] , 
               stripe_by : str | list[str] | Literal[1] | None = None , 
               ignore_cols : list[str] | None = None , stripe_colors = ['#ffffff' , '#a2cffe']):
    int_cols = int_cols or []
    pct_cols = pct_cols or []
    flt_cols = flt_cols or []
    column_definitions = column_definitions or []
    
    if stripe_by and stripe_by != 1:
        if isinstance(stripe_by , str): 
            stripe_by = [stripe_by]
        df0 = df.copy().reset_index()[stripe_by]

    if ignore_cols: 
        df = df.drop(columns=ignore_cols , errors='ignore')
    if int_cols: 
        df = df.assign(**df.loc[:,int_cols].map(lambda x:f'{x:,.0f}'))
    if pct_cols: 
        df = df.assign(**df.loc[:,pct_cols].map(lambda x:f'{x:.{pct_ndigit}%}'))
    if flt_cols: 
        df = df.assign(**df.loc[:,flt_cols].map(lambda x:f'{x:.{flt_ndigit}f}'))
    
    if capitalize:
        df.index.names = [col.title() if isinstance(col , str) else col for col in df.index.names]
        df.columns =     [col.title() if isinstance(col , str) else col for col in df.columns]
        for column_definition in column_definitions:
            column_definition.name = column_definition.name.title()

    column_definitions += [ColumnDefinition(name = str(df.index.names[0]) , width = index_width , 
                                            textprops = {'ha':'left','fontsize':fontsize,'weight':'bold','style':'italic'})]

    for column_definition in column_definitions:
        if column_definition.name not in df.columns and column_definition.name not in df.index.names:
            column_definitions.remove(column_definition)

    tab = Table(df , textprops = {'ha':'center','fontsize':fontsize} ,  
                cell_kw={'edgecolor':'black','linewidth': 0.2,} , 
                column_definitions = column_definitions)
    # tab.col_label_row.set_fontsize(12)
    tab.col_label_row.set_facecolor('b') #'#82cafc'
    #tab.columns[tab.column_names[0]].set_fontsize(12)
    rows = list(tab.rows.values())

    if stripe_by:
        color_i = 1
        for i in range(len(rows)):
            if stripe_by == 1 or i == 0 or (df0.iloc[i] != df0.iloc[i - 1]).any(): 
                color_i = 1 - color_i
            rows[i].set_facecolor(stripe_colors[color_i])
    if emph_last_row: 
        rows[-1].set_facecolor('b') #'#82cafc'
    return tab

def axis_formatter(format : Literal['pct' , 'flt' , 'int' , 'default'] = 'default' , digits = 1):
    if format == 'pct': 
        return FuncFormatter(lambda x,p:f'{x:.{digits}%}')
    elif format == 'flt': 
        return FuncFormatter(lambda x,p:f'{x:.{digits}f}')
    elif format == 'int': 
        return FuncFormatter(lambda x,p:f'{x:d}')
    elif format == 'default': 
        return None

def get_twin_axes(fig : Figure , pos = 111) -> tuple[Axes , Axes]:
    ax1 = fig.add_subplot(pos)
    ax2 : Axes | Any = ax1.twinx()
    return ax1 , ax2

def set_xaxis(ax : Axes , index : Any = None , labels = None , rotation : float | None = 45 , 
              format : Literal['default' , 'pct' , 'flt' , 'int'] = 'default' , digits = 1 , 
              title = '' , title_color = None , 
              tick_pos : Literal['top', 'bottom', 'both', 'default', 'none'] = 'default', 
              tick_color = None , tick_size = None , tick_length = None ,
              num_ticks = 10 , grid = True):
    tick_args : dict[str , Any] = {}
    title_args : dict[str , Any] = {}
    if index is not None:  
        if len(index) == 0: 
            ax.set_xticks([])
        elif labels is not None: 
            assert len(index) == len(labels) , f'len(ticks) {len(index)} != len(labels) {len(labels)}'
            ax.set_xticks(index , labels = labels)
        else:
            ticks = np.unique(index)
            num_ticks = min(num_ticks , len(ticks))
            ticks = ticks[::max(1,len(index)//num_ticks)]
            ax.set_xticks(ticks)

    if rotation is not None:
        ax.xaxis.set_tick_params(rotation=rotation)
    if formatter := axis_formatter(format , digits): 
        ax.xaxis.set_major_formatter(formatter)
    
    if grid: 
        ax.grid()

    if title: 
        title_args['xlabel'] = title 
    if title_color: 
        title_args['color'] = title_color
    if title_args: 
        ax.set_xlabel(**title_args)

    if tick_pos: 
        ax.xaxis.set_ticks_position(tick_pos)

    if tick_size: 
        tick_args['labelsize'] = tick_size
    if tick_length: 
        tick_args['length'] = tick_length
    if tick_color: 
        tick_args['colors'] = tick_color
    # ax.tick_params('x', colors=tick_color) 
    if tick_args: 
        ax.xaxis.set_tick_params(**tick_args)

def set_yaxis(ax : Axes , format : Literal['pct' , 'flt' , 'int'] = 'pct' , digits = 1 , 
              title = '' , title_color = None , 
              tick_pos : Literal['left', 'right', 'both', 'default', 'none'] | None = 'default' , 
              tick_color = None , tick_size = None , tick_length = None , tick_lim = None):
    tick_args : dict[str , Any] = {}
    title_args : dict[str , Any] = {}

    if title: 
        title_args['ylabel'] = title 
    if title_color: 
        title_args['color'] = title_color
    if title_args: 
        ax.set_ylabel(**title_args)

    if formatter := axis_formatter(format , digits): 
        ax.yaxis.set_major_formatter(formatter)

    if tick_pos: 
        ax.yaxis.set_ticks_position(tick_pos)
    if tick_size: 
        tick_args['labelsize'] = tick_size
    if tick_length: 
        tick_args['length'] = tick_length
    if tick_color: 
        tick_args['colors'] = tick_color
    # ax.tick_params('y', colors=tick_color) 
    if tick_args: 
        ax.yaxis.set_tick_params(**tick_args)
    if tick_lim: 
        ax.set_ylim(*tick_lim)

def sns_lineplot(df : pd.DataFrame , x : str , y : str , hue : str , legend : bool = True , legend_loc : str = 'upper left'):
    if isinstance(df[hue].dtype , pd.CategoricalDtype):
        df[hue] = df[hue].cat.remove_unused_categories()
    if CURRENT_SEABORN_VERSION:
        n_hues : int | Any = df[hue].nunique()
        ax = sns.lineplot(x=x, y=y, hue=hue, data=df , palette=sns.diverging_palette(140, 10, sep=10, n=n_hues))
    else:
        ax = sns.lineplot(x=x, y=y, hue=hue, data=df)
    handles, labels = ax.get_legend_handles_labels()
    if legend: 
        ax.legend(handles=handles[0:], labels=labels[0:], loc=legend_loc)
    return ax

def sns_barplot(df : pd.DataFrame , x : str , y : str , hue : str , legend : str | None = 'upper left'):
    if CURRENT_SEABORN_VERSION: 
        n_hues : int | Any = df[hue].nunique()
        ax = sns.barplot(x=x, y=y, hue=hue, data=df , palette=sns.diverging_palette(140, 10, sep=10, n=n_hues))
    else: 
        ax = sns.barplot(x=x, y=y, hue=hue, data=df)
    handles, labels = ax.get_legend_handles_labels()
    if legend: 
        ax.legend(handles=handles[0:], labels=labels[0:], loc=legend)
    return ax
