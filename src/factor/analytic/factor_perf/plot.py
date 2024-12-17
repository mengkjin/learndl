import numpy as np
import pandas as pd
import seaborn as sns

from src.factor.util.plot import plot_table , get_twin_axes , set_xaxis , set_yaxis , PlotMultipleData , PlotFactorData , sns_lineplot , sns_barplot

def plot_factor_frontface(data : pd.DataFrame , show = False):
    with PlotFactorData(data , drop = [] , title = 'Factor Front Face' , show=show) as (df , fig):
        df = df.reset_index().drop(columns=['sum']).set_index('factor_name').sort_values(['factor_name','benchmark']).\
            rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, errors='raise')
        plot_table(df , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                   capitalize=False , stripe_by='factor_name')
    return fig

def plot_factor_coverage(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = ['factor_name'] , title = 'Factor Coverage Ratio' , show=show and i==0) as (df , fig):
            df = df.reset_index().set_index('date')
            ax = sns_lineplot(df , x='date' , y='coverage' , hue='benchmark')

            set_xaxis(ax , df.index.unique() , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title = 'Coverage Ratio' , title_color='b' , tick_lim = (0,1))
            
    return group_plot.fig_dict

def plot_factor_ic_curve(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor IC Curve' , show=show and i==0) as (df , fig):
            df = df.set_index('date').sort_index()
            ax1 , ax2 = get_twin_axes(fig , 111)

            ax1.bar(df.index, df['ic'], color='b', label='IC')
            colors = ['orange','purple','olive','pink','cyan','grey','azure','yellow']
            for col in df.columns.tolist():
                if col.startswith('ma_'): ax1.plot(df.index, df[col], color=colors.pop(0) , label=col)  
            ax1.legend(loc='upper left')

            ax2.plot(df.index, df['cum_ic'], 'r-', label='Cum IC (right)')  
            ax2.legend(loc='upper right')  

            set_xaxis(ax1 , df.index)
            set_yaxis(ax1 , format='flt' , digits=2 , title = 'Period IC' , title_color='b' , tick_color='b')
            set_yaxis(ax2 , format='flt' , digits=2 , title = 'Cummulative IC' , title_color='r' , tick_color='r' , tick_pos=None)
            
    return group_plot.fig_dict

def plot_factor_ic_decay(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = ['factor_name'] , title = 'Factor Average IC Decay' , show=show and i==0) as (df , fig):
            df = df.pivot_table('ic_mean','lag_type','benchmark',observed=False)
            ax = fig.add_subplot(111)
            index_mid = np.arange(len(df)) 
            bar_width = 1 / (len(df.columns) + 1)
            for j , bm in enumerate(df.columns):
                x_pos = index_mid + (j - (len(df.columns) - 1) / 2) * bar_width
                bars = ax.bar(x_pos  , df[bm], label = bm , width=bar_width)  
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2 , height ,
                            f'{height:.4f}' , ha = 'center' , va = 'top' if height < 0 else 'bottom')
            ax.legend(loc='upper right')
            set_xaxis(ax , index_mid , labels = df.index)
            set_yaxis(ax , format='flt' , digits=3 , title = 'Average IC' , title_color='b')

    return group_plot.fig_dict

def plot_factor_ic_indus(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Industry IC & IR' , show=show and i==0) as (df , fig):
            df = df.rename(columns={'avg':'IC_avg','ir':'ICIR'})
            df.sort_values(['IC_avg'], ascending=False, inplace=True)

            ax1 , ax2 = get_twin_axes(fig , 111)
            
            ax1.bar(df['industry'], df['IC_avg'], color='b', alpha=0.5)
            # ax1.spines['bottom'].set_position(('data', 0))
            ax1.legend(['Avg IC'], loc='upper left')

            ax2.plot(df['industry'], df['ICIR'], 'r-')
            ax2.legend(['Avg ICIR'], loc='upper right')

            set_xaxis(ax1 , title = 'Industry')
            set_yaxis(ax1 , format='flt' , digits=2 , title = 'Average IC' , title_color='b')
            set_yaxis(ax2 , format='flt' , digits=2 , title = 'Average ICIR' , title_color='r' , tick_pos=None)
    return group_plot.fig_dict

def plot_factor_ic_year(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Year IC' , show=show and i==0) as (df , fig):
            df['direction'] = 'N(-)' if df['direction'].values[-1] < 0 else 'P(+)'
            df = df.rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, 
                           errors='raise').drop(columns=['sum' , 'year_ret']).rename(columns={'year':'Year'})
            plot_table(df.set_index('Year') , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                       capitalize=False , stripe_by=1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_factor_ic_benchmark(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , drop = ['factor_name'] , title = 'Factor Benchmark IC' , show=show and i==0) as (df , fig):
            df = df.reset_index().drop(columns=['sum']).set_index('benchmark').sort_index().\
            rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)', 'ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, errors='raise')
            plot_table(df , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                       fontsize=10 , capitalize=False , stripe_by=1)
    return group_plot.fig_dict

def plot_factor_ic_monotony(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Percentile Ret & IR' , show=show and i==0) as (df , fig):
            df = df.pivot_table('stats_value' , 'group' , 'stats_name',observed=False).\
                rename(columns={'grp_ret':'RET','grp_ir':'IR'}).reset_index().sort_values('group')
            ax1 , ax2 = get_twin_axes(fig , 111)

            ax1.bar(df['group'], df['RET'], color='b', alpha=0.5)            
            # ax1.spines['bottom'].set_position(('data', 0)) # set position of the bottom spine
            ax1.legend(['Avg Ret'], loc='upper left')

            ax2.plot(df['group'], df['IR'], 'r-')
            ax2.legend(['Avg IR'], loc='upper right')

            set_xaxis(ax1 , index = [] , tick_pos='bottom')
            set_yaxis(ax1 , format='pct' , digits=2 , title = 'Average Ret' , title_color='b')
            set_yaxis(ax2 , format='flt' , digits=3 , title = 'Average IR' , title_color='r' , tick_pos=None)
    return group_plot.fig_dict

def plot_factor_pnl_curve(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Cummulative Long-Short PnL' , show=show and i==0) as (df , fig):
            df = df.pivot_table(index='date' , columns='weight_type' , values='cum_ret')
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[weight_type], label=weight_type) for weight_type in df.columns.tolist()]
            ax.legend()

            set_xaxis(ax , df.index , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative PnL' , title_color='b')
            
    return group_plot.fig_dict

def plot_factor_style_corr(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Corr Curve with Risk Styles' , show=show and i==0) as (df , fig):
            df = df.set_index('date').sort_index()
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[style], label=style) for style in df.columns]
            
            set_xaxis(ax , df.index , title = 'Trade Date')
            set_yaxis(ax , format='flt' , digits=2 , title = 'Factor / Style Correlation' , title_color='b')

    return group_plot.fig_dict

def plot_factor_style_corr_distrib(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Corr Distribution with Risk Styles' , show=show and i==0) as (df , fig):
            df = df.set_index(['date']).sort_index()
            df.columns.rename('style_factor', inplace=True)
            df = df.stack().rename('factor_corr').reset_index(drop=False) # type: ignore
            ax = sns.boxplot(x='style_factor', y='factor_corr', data=df, width=0.3)

            set_xaxis(ax , title = 'Trade Date')
            set_yaxis(ax , format='flt' , digits=2 , title = 'Factor / Style Correlation' , title_color='b')
            
    return group_plot.fig_dict

def plot_factor_group_curve(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Group CumReturn' , show=show and i==0) as (df , fig):
            df = df.set_index('date')
            ax = sns_lineplot(df , x='date' , y='cum_ret' , hue='group')

            set_xaxis(ax , df.index.unique() , title = 'Trade Date')
            set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b')

    return group_plot.fig_dict

def plot_factor_group_decay(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Groups Return Decay' , show=show and i==0) as (df , fig):
            df = df[df['stats_name'] == 'decay_grp_ret']

            ax = sns_barplot(df , x='lag_type' , y='stats_value' , hue='group')
            set_xaxis(ax , title = 'Lag Type')
            set_yaxis(ax , format = 'pct' , digits=2 , title = 'Groups Average Return' , title_color='b')
            
    return group_plot.fig_dict

def plot_factor_group_ir_decay(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Groups IR Decay' , show=show and i==0) as (df , fig):
            df = df[df['stats_name'] == 'decay_grp_ir']
            ax = sns_barplot(df , x='lag_type' , y='stats_value' , hue='group')

            set_xaxis(ax , title = 'Lag Type')
            set_yaxis(ax , format = 'flt' , digits=3 , title = 'Groups Average IR' , title_color='b')

    return group_plot.fig_dict

def plot_factor_group_year(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Annualized Top Group Performance' , show=show and i==0) as (df , fig):
            df = df.drop(columns=['abs_avg'])
            plot_table(df.set_index('year') , 
                       int_cols = ['group'] ,
                       pct_cols = ['sum' , 'avg' , 'year_ret', 'std', 'cum_mdd'] , 
                       flt_cols = ['ir'] , pct_ndigit = 3 , stripe_by=1 , emph_last_row=True)
    return group_plot.fig_dict

def plot_factor_distrib_curve(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Cross-Sectional Distribution' , show=show and i==0 , suptitle=True) as (df , fig):
                assert not df['date'].duplicated().any() , df['date']
                lay_out = (3, min(int(np.ceil(len(df) / 3)) , 3))
                for i in range(min(len(df) , lay_out[0] * lay_out[1])):
                    ax = fig.add_subplot(*lay_out , i + 1)
                    day_df = df.iloc[i]
                    bins = day_df['hist_bins']
                    cnts = day_df['hist_cnts'] / day_df['hist_cnts'].sum()
                    ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), color='b' , alpha = 0.5)
                    set_yaxis(ax , format='pct' , digits=0 , tick_size=8 , tick_length=0)
                    set_xaxis(ax , format='flt' , digits=1 , tick_size=8 , tick_length=0 , rotation = None)
                    ax.set_title(str(day_df['date']))
    return group_plot.fig_dict

def plot_factor_distrib_qtile(data : pd.DataFrame , show = False):
    group_plot = PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
    for i , sub_data in enumerate(group_plot):     
        with PlotFactorData(sub_data , title = 'Factor Cross-Sectional Quantile' , show=show and i==0) as (df , fig):
            df.columns.rename('quantile_name', inplace=True)
            df = df.set_index('date').stack().rename('quantile_value').reset_index().set_index('date') # type: ignore
            ax = sns_lineplot(df , x='date' , y='quantile_value' , hue='quantile_name')

            set_xaxis(ax , df.index , title = 'Trade Date')
            set_yaxis(ax , format='flt' , digits=2 , title = 'Quantile Value' , title_color='b')
            
    return group_plot.fig_dict