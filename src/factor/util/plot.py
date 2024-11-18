import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from packaging import version
from plottable import Table , ColumnDefinition
from typing import Any , Callable , Optional

from ...basic.conf import CATEGORIES_BENCHMARKS

CURRENT_SEABORN_VERSION = version.Version(getattr(sns , '__version__')) > version.Version('0.9.1')

sns.set_theme(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.family'] = ['monospace'] # ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei'] # for chinese
plt.rcParams['axes.unicode_minus'] = False

def bm_order(bm_list):
    new_bm_list = [bm for bm in CATEGORIES_BENCHMARKS if bm in bm_list]
    return new_bm_list

def bm_name(bm : Any | str | None = None):
    if bm is None or bm == '': name = 'none'
    elif isinstance(bm ,str) : name = bm
    else: name = bm.name
    assert isinstance(name , str) , bm
    return name

def multi_factor_plot(func : Callable):
    def wrapper(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , 
                **kwargs) -> dict[str,Figure]:
        factor_list = [factor_name] if factor_name else df['factor_name'].unique()
        if benchmark or ('benchmark' not in df.columns):
            bench_list = [benchmark] 
        elif 'benchmark' in df.columns:
            bench_list = bm_order(df['benchmark'].unique())
        return {f'{fn}.{bm_name(bn)}':func(df , factor_name = fn , benchmark = bm_name(bn) , **kwargs) for fn in factor_list for bn in bench_list}
    return wrapper

def plot_df_filter(df : pd.DataFrame , factor_name : str | Any , benchmark : str) -> pd.DataFrame:
    #assert factor_name is not None and (factor_name in df['factor_name'].values) , factor_name
    if df.index.name: df = df.reset_index() 
    if 'factor_name' in df.columns and factor_name is not None:
        assert isinstance(factor_name , str) and factor_name in df['factor_name'].values , (factor_name , df)
        df = df[df['factor_name'] == factor_name].drop(columns=['factor_name'])
    if 'benchmark' in df.columns and benchmark is not None:
        assert isinstance(benchmark , str) and benchmark in df['benchmark'].values , (benchmark , df)
        df = df[df['benchmark'] == benchmark].drop(columns=['benchmark'])
    return df

def plot_head(df : pd.DataFrame , factor_name : str | Any , benchmark : str | Any = None) -> tuple[pd.DataFrame , Figure]:
    df = plot_df_filter(df , factor_name , benchmark)
    for col in df.columns: 
        if col.endswith('date'):
            df[col] = df[col].astype(str)
    fig = plt.figure(figsize=(16, 7))
    return df , fig

def plot_tail(title_head : str , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False , suptitle = False):
    title = f'{title_head} for [{factor_name}]'
    if benchmark: title += f' in {benchmark}'
    plt.suptitle(title , fontsize = 14) if suptitle else plt.title(title , fontsize = 14)
    if not show: plt.close()

def plot_table(df : pd.DataFrame , pct_cols = [] , flt_cols = [] , capitalize = True , 
               pct_ndigit = 2 , flt_ndigit = 3 , emph_last_row = True , 
               stripe_rows : int | list[int] = 1 , column_definitions = []):
    if pct_cols: df = df.assign(**df.loc[:,pct_cols].map(lambda x:f'{x:.{pct_ndigit}%}'))
    if flt_cols: df = df.assign(**df.loc[:,flt_cols].map(lambda x:f'{x:.{flt_ndigit}f}'))
    if capitalize:
        df.index.names = [col.capitalize() if isinstance(col , str) else col for col in df.index.names]
        df.columns = [col.capitalize() if isinstance(col , str) else col for col in df.columns]
    column_definitions += [ColumnDefinition(name = df.index.names[0] , 
                                            textprops = {'ha':'left','fontsize':10,'weight':'bold','style':'italic'})]
    tab = Table(df , textprops = {'ha':'center','fontsize':10} ,  
                cell_kw={'edgecolor':'black','linewidth': 0.2,} , 
                column_definitions = column_definitions)
    # tab.col_label_row.set_fontsize(12)
    tab.col_label_row.set_facecolor('b') #'#82cafc'
    #tab.columns[tab.column_names[0]].set_fontsize(12)
    rows = list(tab.rows.values())
    if stripe_rows: 
        if isinstance(stripe_rows , int):
            stripe_rows = [stripe_rows] * (int(len(rows) / stripe_rows) + 1)
        row_i , row_color = 0 , False
        while stripe_rows and row_i < len(rows):
            r = stripe_rows.pop(0)
            if row_color: [r.set_facecolor('#a2cffe') for r in rows[row_i:row_i + r]] # '#d5ffff'
            row_color = not row_color
            row_i += r
    if emph_last_row: rows[-1].set_facecolor('b') #'#82cafc'
    return tab

def plot_xaxis(ax : Axes , index : Any = None , num_ticks : int = 10):
    if index is not None:  ax.set_xticks(index[::max(1,len(index)//num_ticks)])
    ax.grid()
    ax.xaxis.set_tick_params(rotation=45)