import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plottable import ColumnDefinition, ColDef, Table
from plottable.formatters import decimal_to_percent
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from packaging import version
from typing import Any , Callable , Literal , Optional

CURRENT_SEABORN_VERSION = version.Version(getattr(sns , '__version__')) > version.Version('0.9.1')

sns.set_theme(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def pct_fmt(temp : float, position : int = 2): return f'{temp:.2%}'
def pct_d2f(temp : Any, position : int = 2): return f'{temp:.2f}'
def bm_name(bm : Any | str | None = None):
    if bm is None or bm == '': name = 'default'
    elif isinstance(bm ,str) : name = bm
    else: name = bm.name
    assert isinstance(name , str) , bm
    return name

def multi_factor_plot(func : Callable):
    def wrapper(df : pd.DataFrame , *args , factor_name : Optional[str] = None , benchmark : Optional[str] = None , 
                **kwargs) -> dict[str,dict[str,Figure]]:
        factor_list = [factor_name] if factor_name else df['factor_name'].unique()
        bench_list = [benchmark] if benchmark else df['benchmark'].unique() if 'benchmark' in df.columns else [None]
        return {fn:{bm_name(bn):func(df , *args , factor_name = fn , benchmark = bm_name(bn) , **kwargs) for bn in bench_list} for fn in factor_list}
    return wrapper

def plot_df_filter(df : pd.DataFrame , factor_name : str | Any , benchmark : str) -> pd.DataFrame:
    assert factor_name is not None and (factor_name in df['factor_name'].values) , factor_name
    if df.index.name: df = df.reset_index() 
    df = df[df['factor_name'] == factor_name].drop(columns=['factor_name'])
    if 'benchmark' in df.columns:
        assert isinstance(benchmark , str) and benchmark in df['benchmark'].values , (benchmark , df)
        df = df[df['benchmark'] == benchmark].drop(columns=['benchmark'])
    return df

def plot_head(df : pd.DataFrame , factor_name : str | Any , benchmark : str | Any = None) -> tuple[pd.DataFrame , Figure]:
    df = plot_df_filter(df , factor_name , benchmark)
    [df.assign(**{col:df[col].astype(str)}) for col in df.columns]
    fig = plt.figure(figsize=(16, 7))
    return df , fig

def plot_tail(title_head : str , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False , suptitle = False):
    title = f'{title_head} for [{factor_name}]'
    if benchmark: title += f' in {benchmark}'
    plt.suptitle(title , fontsize = 14) if suptitle else plt.title(title , fontsize = 14)
    plt.xticks(rotation=45)  
    if not show: plt.close()

def plot_table(df , **kwargs):
    tab = Table(df , textprops = {'ha':'center','fontsize':10,'weight':'bold'} , 
                odd_row_color='aliceblue' , 
                cell_kw={'edgecolor':'black','linewidth': 0.2,} , **kwargs)
    tab.col_label_row.set_fontsize(12)
    tab.col_label_row.set_facecolor('lightskyblue')
    tab.columns[tab.column_names[0]].set_fontsize(12)
    list(tab.rows.values())[-1].set_facecolor('lightskyblue')
    return tab