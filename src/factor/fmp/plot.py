import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plottable import ColumnDefinition, ColDef
from plottable.formatters import decimal_to_percent
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from typing import Any , Callable , Literal , Optional

from ..basic.plot import pct_fmt , pct_d2f, multi_factor_plot , plot_head , plot_tail , plot_table

@multi_factor_plot
def plot_lag_perf_curve(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)

    df = df.set_index('trade_date')
    for col in df.columns: 
        plt.plot(df.index , df[col], label=f'{factor_name}.{benchmark}.{col}')

    plt.grid()
    plt.legend(loc = 'upper left')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    plot_tail(f'Cumulative Excess Return' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_perf_yearly(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover']
    flt_cols = ['ir','calmar']
    df = df.assign(**df.loc[:,pct_cols].map(lambda x:f'{x:.2%}') , **df.loc[:,flt_cols].map(lambda x:f'{x:.3f}'))
    df.columns = [col.capitalize() for col in df.columns]
    tab = plot_table(df.set_index('Year') , column_definitions = [ColumnDefinition(name='mdd_period', width=2)])
    plot_tail(f'Yearly Performance' , factor_name , benchmark , show , suptitle = False)
    return fig

@multi_factor_plot
def plot_fmp_perf_monthly(df : pd.DataFrame , factor_name : Optional[str] = None , benchmark : Optional[str] = None , show = False):
    df , fig = plot_head(df , factor_name , benchmark)
    pct_cols = ['pf','bm','excess','annualized','mdd','te','turnover']
    flt_cols = ['ir','calmar']
    df = df.assign(**df.loc[:,pct_cols].map(lambda x:f'{x:.2%}') , **df.loc[:,flt_cols].map(lambda x:f'{x:.3f}'))
    df.columns = [col.capitalize() for col in df.columns]

    plot_table(df.set_index('Month'))

    plot_tail(f'Month Performance' , factor_name , benchmark , show , suptitle = False)
    return fig