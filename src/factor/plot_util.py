import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from packaging import version
from typing import Literal , Optional

# for Chinese
sns.set_theme(context='notebook', style='ticks', font='SimHei', rc={'axes.unicode_minus': False})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_SEABORN_VERSION = version.Version(getattr(sns , '__version__')) > version.Version('0.9.1')


class FactorStatPlot:

    @staticmethod
    def pct_fmt(temp : float, position : int = 2): return f'{np.format_float_positional(temp * 100 , position)}%'

    @classmethod
    def plot_decay_ic(cls , ic_lag_df : pd.DataFrame):
        fig = plt.figure(figsize=(16, 7))

        x_labels = ic_lag_df['lag_type'].unique().tolist()
        factors  = ic_lag_df['factor_name'].unique().tolist()
        
        bar_width , spacing = 0.2 , 0.05  
        total_width = (bar_width + spacing) * (len(factors)  - 1) + bar_width  
        offsets = np.linspace(0, (len(factors)  - 1) * (bar_width + spacing), len(factors) ) - (total_width / 2)  

        for i , (factor_name, group) in enumerate(ic_lag_df.groupby('factor_name')):  
            
            x = np.arange(len(x_labels)) + offsets[i]  
            bars = plt.bar(x, group['ic_mean'], width=bar_width,  label=factor_name) 
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2 , height ,
                         f'{height:.4f}' , ha = 'center' , va = 'top' if height < 0 else 'bottom')

        plt.xticks(range(len(x_labels)), x_labels, rotation=45)  
        plt.legend()  
        plt.title('Average IC Decay')
        
        return fig
    
    @classmethod
    def plot_decay_grp_perf(cls , grp_perf_lag : pd.DataFrame , stat_type : Literal['ret' , 'ir'] = 'ret',  factor_name : Optional[str] = None):
        if factor_name is None:
            factor_list = grp_perf_lag['factor_name'].unique()
            return [cls.plot_decay_grp_perf(grp_perf_lag , stat_type , factor_name) for factor_name in factor_list]
        else:
            factor_grp_perf = grp_perf_lag[(grp_perf_lag['factor_name'] == factor_name) & 
                                        (grp_perf_lag['stats_name'] == f'decay_grp_{stat_type}')].drop(columns=['factor_name'])
            group_num = factor_grp_perf['group'].nunique()
            fig = plt.figure(figsize=(16, 7))
            if CURRENT_SEABORN_VERSION:
                ax = sns.barplot(x='lag_type', y='stats_value', hue='group', data=factor_grp_perf,
                                palette=sns.diverging_palette(140, 10, sep=10, n=group_num))
            else:
                ax = sns.barplot(x='lag_type', y='stats_value', hue='group', data=factor_grp_perf)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[0:], labels=labels[0:])
            ax.set_title(f'Groups {stat_type.upper()} Decay for {factor_name}')
            plt.xticks(rotation=45)  
            ax.legend(loc='upper left')
            return fig
        
    @classmethod
    def plot_group_perf(cls , grp_perf : pd.DataFrame , factor_name : Optional[str] = None):
        if factor_name is None:
            factor_list = grp_perf['factor_name'].unique()
            return [cls.plot_group_perf(grp_perf , factor_name) for factor_name in factor_list]
        else:
            grp_cum_ret = grp_perf[grp_perf['factor_name'] == factor_name].drop(columns=['factor_name']).\
                set_index(['date', 'group']).groupby('group' , observed=False)['group_ret'].\
                cumsum().rename('cum_ret').reset_index(drop=False)
            grp_cum_ret = grp_cum_ret.assign(date=pd.to_datetime(grp_cum_ret['date'].astype(str), format='%Y%m%d'))
            fig = plt.figure(figsize=(16, 7))
            if CURRENT_SEABORN_VERSION:
                ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=grp_cum_ret,
                                  palette=sns.diverging_palette(140, 10, sep=10, n=grp_cum_ret['group'].nunique()))
            else:
                ax = sns.lineplot(x='date', y='cum_ret', hue='group', data=grp_cum_ret)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[0:], labels=labels[0:], loc='upper left')
            plt.grid()
            plt.gca().yaxis.set_major_formatter(FuncFormatter(cls.pct_fmt))
            plt.xticks(rotation=45)  
            ax.set_title(f'Group CumReturn for {factor_name}')
            return fig
        
    @classmethod
    def plot_style_corr(cls , style_corr : pd.DataFrame , factor_name : Optional[str] = None):
        if factor_name is None:
            factor_list = style_corr['factor_name'].unique()
            return [cls.plot_style_corr(style_corr , factor_name) for factor_name in factor_list]
        else:
            factor_style_corr = style_corr[style_corr['factor_name'] == factor_name].drop(columns=['factor_name']).set_index(['date'])
            x_labels = factor_style_corr.columns.tolist()
            factor_style_corr.columns.rename('style_factor', inplace=True)
            factor_style_corr = factor_style_corr.stack().rename('factor_corr').reset_index(drop=False) # type: ignore
            fig = plt.figure(figsize=(16, 7))
            ax = sns.boxplot(x='style_factor', y='factor_corr', data=factor_style_corr, width=0.3)
            plt.grid()
            plt.xticks(rotation=45)  
            ax.set_title(f'Correlation with Risk Style Factors for {factor_name}')
            return fig