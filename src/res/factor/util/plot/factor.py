import numpy as np
import pandas as pd
import seaborn as sns

import src.math.plot as plot

from .plot_basic import PlotDfFigIterator

class Plotter:
    def __init__(self , title_prefix = 'Factor'):
        self.plot_iter = PlotDfFigIterator(title_prefix)

    def plot_frontface(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Front Face' , [] , drop_keys = False , drop_cols = [])
        for df , fig in self.plot_iter.iter():
            df = df.reset_index().set_index('factor_name').sort_values(['factor_name','benchmark'])
            plot.plot_table(df , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , '|IC|_avg'] , capitalize=False , stripe_by='factor_name')
        return self.plot_iter.figs

    def plot_coverage(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Coverage Ratio' , ['factor_name'])
        for df , fig in self.plot_iter.iter():
            df = df.reset_index().set_index('date')
            ax = plot.sns_lineplot(df , x='date' , y='coverage' , hue='benchmark')

            plot.set_xaxis(ax , df.index.unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Coverage Ratio' , title_color='b' , tick_lim = (0.,1.2))
                
        return self.plot_iter.figs

    def plot_ic_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'IC Curve' , ['factor_name' , 'benchmark'] , dropna = 'all')
        for df , fig in self.plot_iter.iter():
            df = df.set_index('date').sort_index()
            ax1 , ax2 = plot.get_twin_axes(fig , 111)

            ax1.bar(df.index, df['ic'], color='b', label='IC')
            colors = ['orange','purple','olive','pink','cyan','grey','azure','yellow']
            for col in df.columns.tolist():
                if col.startswith('ma_'): 
                    ax1.plot(df.index, df[col], color=colors.pop(0) , label=col)  
            ax1.legend(loc='upper left')

            ax2.plot(df.index, df['cum_ic'], 'r-', label='Cum IC (right)')  
            ax2.legend(loc='upper right')  

            plot.set_xaxis(ax1 , df.index , title = 'Trade Date')
            plot.set_yaxis(ax1 , format='flt' , digits=2 , title = 'Period IC' , title_color='b' , tick_color='b')
            plot.set_yaxis(ax2 , format='flt' , digits=2 , title = 'Cummulative IC' , title_color='r' , tick_color='r' , tick_pos=None)
                
        return self.plot_iter.figs

    def plot_ic_decay(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Average IC Decay' , ['factor_name'])
        for df , fig in self.plot_iter.iter():
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
            plot.set_xaxis(ax , index_mid , labels = df.index)
            plot.set_yaxis(ax , format='flt' , digits=3 , title = 'Average IC' , title_color='b')
        return self.plot_iter.figs

    def plot_ic_indus(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Industry IC & IR' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.rename(columns={'avg':'IC_avg','ir':'ICIR'})
            df.sort_values(['IC_avg'], ascending=False, inplace=True)

            ax1 , ax2 = plot.get_twin_axes(fig , 111)
            
            ax1.bar(df['industry'], df['IC_avg'], color='b', alpha=0.5)
            # ax1.spines['bottom'].set_position(('data', 0))
            ax1.legend(['Avg IC'], loc='upper left')

            ax2.plot(df['industry'], df['ICIR'], 'r-')
            ax2.legend(['Avg ICIR'], loc='upper right')

            plot.set_xaxis(ax1 , title = 'Industry')
            plot.set_yaxis(ax1 , format='flt' , digits=2 , title = 'Average IC' , title_color='b')
            plot.set_yaxis(ax2 , format='flt' , digits=2 , title = 'Average ICIR' , title_color='r' , tick_pos=None)
        return self.plot_iter.figs

    def plot_ic_year(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , '`Industr`y IC & IR' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df['direction'] = 'N(-)' if df['direction'].values[-1] < 0 else 'P(+)'
            df = df.rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, 
                        errors='raise').drop(columns=['sum']).rename(columns={'year':'Year'})
            plot.plot_table(df.set_index('Year') , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                    capitalize=False , stripe_by=1 , emph_last_row=True)
        return self.plot_iter.figs

    def plot_ic_benchmark(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Benchmark IC' , ['factor_name'])
        for df , fig in self.plot_iter.iter():
            df = df.reset_index().drop(columns=['sum']).set_index('benchmark').sort_index().\
            rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)', 'ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, errors='raise')
            plot.plot_table(df , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                            fontsize=10 , capitalize=False , stripe_by=1)
        return self.plot_iter.figs

    def plot_ic_monotony(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Percentile Ret & IR' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.pivot_table('stats_value' , 'group' , 'stats_name',observed=False).\
                rename(columns={'grp_ret':'RET','grp_ir':'IR'}).reset_index().sort_values('group')
            ax1 , ax2 = plot.get_twin_axes(fig , 111)

            ax1.bar(df['group'], df['RET'], color='b', alpha=0.5)            
            # ax1.spines['bottom'].set_position(('data', 0)) # set position of the bottom spine
            ax1.legend(['Avg Ret'], loc='upper left')

            ax2.plot(df['group'], df['IR'], 'r-')
            ax2.legend(['Avg IR'], loc='upper right')

            plot.set_xaxis(ax1 , index = [] , tick_pos='bottom')
            plot.set_yaxis(ax1 , format='pct' , digits=2 , title = 'Average Ret' , title_color='b')
            plot.set_yaxis(ax2 , format='flt' , digits=3 , title = 'Average IR' , title_color='r' , tick_pos=None)
        return self.plot_iter.figs

    def plot_pnl_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Cummulative Long-Short PnL' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.pivot_table(index='date' , columns='weight_type' , values='cum_ret')
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[weight_type], label=weight_type) for weight_type in df.columns.tolist()]
            ax.legend()

            plot.set_xaxis(ax , df.index , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative PnL' , title_color='b')
                
        return self.plot_iter.figs

    def plot_style_corr(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Corr Curve with Risk Styles' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.set_index('date').sort_index()
            ax = fig.add_subplot(111)
            [ax.plot(df.index , df[style], label=style) for style in df.columns]
            
            plot.set_xaxis(ax , df.index , title = 'Trade Date')
            plot.set_yaxis(ax , format='flt' , digits=2 , title = 'Factor / Style Correlation' , title_color='b')

        return self.plot_iter.figs

    def plot_style_corr_distrib(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Corr Distribution with Risk Styles' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.set_index(['date']).sort_index()
            df.columns.rename('style_factor', inplace=True)
            df = df.stack().rename('factor_corr').reset_index(drop=False) # type: ignore
            ax = sns.boxplot(x='style_factor', y='factor_corr', data=df, width=0.3)

            plot.set_xaxis(ax , title = 'Trade Date')
            plot.set_yaxis(ax , format='flt' , digits=2 , title = 'Factor / Style Correlation' , title_color='b')
            
        return self.plot_iter.figs

    def plot_group_return(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Group Excess Return' , ['factor_name'])
        for df , fig in self.plot_iter.iter():
            df = df.transpose()
            ax = fig.add_subplot(111)
            index_mid = np.arange(len(df)) 
            bar_width = 1 / (len(df.columns) + 1)
            for j , bm in enumerate(df.columns):
                x_pos = index_mid + (j - (len(df.columns) - 1) / 2) * bar_width
                ax.bar(x_pos  , df[bm], label = bm , width=bar_width)  
            ax.legend(loc='upper right')
            plot.set_xaxis(ax , index_mid , labels = df.index)
            plot.set_yaxis(ax , format='pct' , digits=3 , title = 'Group Excess Return' , title_color='b')
        return self.plot_iter.figs

    def plot_group_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Group Cummulative Return' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.set_index('date')
            ax = plot.sns_lineplot(df , x='date' , y='cum_ret' , hue='group')

            plot.set_xaxis(ax , df.index.unique() , title = 'Trade Date')
            plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b')

        return self.plot_iter.figs

    def plot_group_decay(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Groups Return Decay' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.query('stats_name == "decay_grp_ret"')
            ax = plot.sns_barplot(df , x='lag_type' , y='stats_value' , hue='group')
            plot.set_xaxis(ax , title = 'Lag Type')
            plot.set_yaxis(ax , format = 'pct' , digits=2 , title = 'Groups Average Return' , title_color='b')
                
        return self.plot_iter.figs

    def plot_group_ir_decay(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Groups IR Decay' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.query('stats_name == "decay_grp_ir"')
            ax = plot.sns_barplot(df , x='lag_type' , y='stats_value' , hue='group')
            plot.set_xaxis(ax , title = 'Lag Type')
            plot.set_yaxis(ax , format = 'flt' , digits=3 , title = 'Groups Average IR' , title_color='b')

        return self.plot_iter.figs

    def plot_group_year(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Annualized Top Group Performance' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df = df.drop(columns=['abs_avg'])
            plot.plot_table(df.set_index('year') , 
                            int_cols = ['group'] ,
                            pct_cols = ['sum' , 'avg' , 'year_ret', 'std', 'cum_mdd'] , 
                            flt_cols = ['ir'] , pct_ndigit = 3 , stripe_by=1 , emph_last_row=True)
        return self.plot_iter.figs

    def plot_distrib_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Cross-Sectional Distribution' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            assert not df['date'].duplicated().any() , df['date']
            lay_out = (3, min(int(np.ceil(len(df) / 3)) , 3))
            for i in range(min(len(df) , lay_out[0] * lay_out[1])):
                ax = fig.add_subplot(*lay_out , i + 1)
                day_df = df.iloc[i]
                bins = day_df['hist_bins']
                cnts = day_df['hist_cnts'] / day_df['hist_cnts'].sum()
                ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), color='b' , alpha = 0.5)
                plot.set_yaxis(ax , format='pct' , digits=0 , tick_size=8 , tick_length=0)
                plot.set_xaxis(ax , format='flt' , digits=1 , tick_size=8 , tick_length=0 , rotation = None)
                ax.set_title(str(day_df['date']))
        return self.plot_iter.figs

    def plot_distrib_qtile(self , data : pd.DataFrame , show = False , title_prefix = None):
        self.plot_iter.set_args(data , show , title_prefix , 'Cross-Sectional Quantile' , ['factor_name' , 'benchmark'])
        for df , fig in self.plot_iter.iter():
            df.columns.rename('quantile_name', inplace=True)
            df = df.set_index('date').stack().rename('quantile_value').reset_index().set_index('date') # type: ignore
            ax = plot.sns_lineplot(df , x='date' , y='quantile_value' , hue='quantile_name')

            plot.set_xaxis(ax , df.index , title = 'Trade Date')
            plot.set_yaxis(ax , format='flt' , digits=2 , title = 'Quantile Value' , title_color='b')
                
        return self.plot_iter.figs

    """
    def plot_ic_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} IC Curve'
        for i , sub_data in enumerate(group_plot):  
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0 , dropna = 'all') as (df , fig):
                df = df.set_index('date').sort_index()
                ax1 , ax2 = plot.get_twin_axes(fig , 111)

                ax1.bar(df.index, df['ic'], color='b', label='IC')
                colors = ['orange','purple','olive','pink','cyan','grey','azure','yellow']
                for col in df.columns.tolist():
                    if col.startswith('ma_'): 
                        ax1.plot(df.index, df[col], color=colors.pop(0) , label=col)  
                ax1.legend(loc='upper left')

                ax2.plot(df.index, df['cum_ic'], 'r-', label='Cum IC (right)')  
                ax2.legend(loc='upper right')  

                plot.set_xaxis(ax1 , df.index , title = 'Trade Date')
                plot.set_yaxis(ax1 , format='flt' , digits=2 , title = 'Period IC' , title_color='b' , tick_color='b')
                plot.set_yaxis(ax2 , format='flt' , digits=2 , title = 'Cummulative IC' , title_color='r' , tick_color='r' , tick_pos=None)
                
        return group_plot.fig_dict

    def plot_frontface(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = [])
        title = f'{title_prefix or self.title_prefix} Front Face'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , drop = [] , title = title , show=show and i==0) as (df , fig):
                df = df.reset_index().drop(columns=['sum']).set_index('factor_name').sort_values(['factor_name','benchmark']).\
                    rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, errors='raise')
                
                plot.plot_table(df , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                                capitalize=False , stripe_by='factor_name')
        return group_plot.fig_dict

    def plot_coverage(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name'])
        title = f'{title_prefix or self.title_prefix} Coverage Ratio'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , drop = ['factor_name'] , title = title , show=show and i==0) as (df , fig):
                df = df.reset_index().set_index('date')
                ax = plot.sns_lineplot(df , x='date' , y='coverage' , hue='benchmark')

                plot.set_xaxis(ax , df.index.unique() , title = 'Trade Date')
                plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Coverage Ratio' , title_color='b' , tick_lim = (0.,1.2))
                
        return group_plot.fig_dict

    def plot_ic_decay(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name'])
        title = f'{title_prefix or self.title_prefix} Average IC Decay'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , drop = ['factor_name'] , title = title , show=show and i==0) as (df , fig):
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
                plot.set_xaxis(ax , index_mid , labels = df.index)
                plot.set_yaxis(ax , format='flt' , digits=3 , title = 'Average IC' , title_color='b')

        return group_plot.fig_dict

    def plot_ic_indus(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Industry IC & IR'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.rename(columns={'avg':'IC_avg','ir':'ICIR'})
                df.sort_values(['IC_avg'], ascending=False, inplace=True)

                ax1 , ax2 = plot.get_twin_axes(fig , 111)
                
                ax1.bar(df['industry'], df['IC_avg'], color='b', alpha=0.5)
                # ax1.spines['bottom'].set_position(('data', 0))
                ax1.legend(['Avg IC'], loc='upper left')

                ax2.plot(df['industry'], df['ICIR'], 'r-')
                ax2.legend(['Avg ICIR'], loc='upper right')

                plot.set_xaxis(ax1 , title = 'Industry')
                plot.set_yaxis(ax1 , format='flt' , digits=2 , title = 'Average IC' , title_color='b')
                plot.set_yaxis(ax2 , format='flt' , digits=2 , title = 'Average ICIR' , title_color='r' , tick_pos=None)
        return group_plot.fig_dict

    def plot_ic_year(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Year IC'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df['direction'] = 'N(-)' if df['direction'].values[-1] < 0 else 'P(+)'
                df = df.rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)','ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, 
                            errors='raise').drop(columns=['sum']).rename(columns={'year':'Year'})
                plot.plot_table(df.set_index('Year') , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                        capitalize=False , stripe_by=1 , emph_last_row=True)
        return group_plot.fig_dict

    def plot_ic_benchmark(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name'])
        title = f'{title_prefix or self.title_prefix} Benchmark IC'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , drop = ['factor_name'] , title = title , show=show and i==0) as (df , fig):
                df = df.reset_index().drop(columns=['sum']).set_index('benchmark').sort_index().\
                rename(columns={'avg': 'IC_avg', 'std': 'IC_std','year_ret':'IC(ann)', 'ir': 'ICIR','abs_avg' :'abs(IC)_avg' , 'cum_mdd': 'IC_mdd'}, errors='raise')
                plot.plot_table(df , flt_cols = ['IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , 'abs(IC)_avg'] , 
                                fontsize=10 , capitalize=False , stripe_by=1)
        return group_plot.fig_dict

    def plot_ic_monotony(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Percentile Ret & IR'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.pivot_table('stats_value' , 'group' , 'stats_name',observed=False).\
                    rename(columns={'grp_ret':'RET','grp_ir':'IR'}).reset_index().sort_values('group')
                ax1 , ax2 = plot.get_twin_axes(fig , 111)

                ax1.bar(df['group'], df['RET'], color='b', alpha=0.5)            
                # ax1.spines['bottom'].set_position(('data', 0)) # set position of the bottom spine
                ax1.legend(['Avg Ret'], loc='upper left')

                ax2.plot(df['group'], df['IR'], 'r-')
                ax2.legend(['Avg IR'], loc='upper right')

                plot.set_xaxis(ax1 , index = [] , tick_pos='bottom')
                plot.set_yaxis(ax1 , format='pct' , digits=2 , title = 'Average Ret' , title_color='b')
                plot.set_yaxis(ax2 , format='flt' , digits=3 , title = 'Average IR' , title_color='r' , tick_pos=None)
        return group_plot.fig_dict

    def plot_pnl_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Cummulative Long-Short PnL'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.pivot_table(index='date' , columns='weight_type' , values='cum_ret')
                ax = fig.add_subplot(111)
                [ax.plot(df.index , df[weight_type], label=weight_type) for weight_type in df.columns.tolist()]
                ax.legend()

                plot.set_xaxis(ax , df.index , title = 'Trade Date')
                plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative PnL' , title_color='b')
                
        return group_plot.fig_dict

    def plot_style_corr(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Corr Curve with Risk Styles'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.set_index('date').sort_index()
                ax = fig.add_subplot(111)
                [ax.plot(df.index , df[style], label=style) for style in df.columns]
                
                plot.set_xaxis(ax , df.index , title = 'Trade Date')
                plot.set_yaxis(ax , format='flt' , digits=2 , title = 'Factor / Style Correlation' , title_color='b')

        return group_plot.fig_dict

    def plot_style_corr_distrib(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Corr Distribution with Risk Styles'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.set_index(['date']).sort_index()
                df.columns.rename('style_factor', inplace=True)
                df = df.stack().rename('factor_corr').reset_index(drop=False) # type: ignore
                ax = sns.boxplot(x='style_factor', y='factor_corr', data=df, width=0.3)

                plot.set_xaxis(ax , title = 'Trade Date')
                plot.set_yaxis(ax , format='flt' , digits=2 , title = 'Factor / Style Correlation' , title_color='b')
                
        return group_plot.fig_dict

    def plot_group_return(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name'])
        
        title = f'{title_prefix or self.title_prefix} Group Excess Return'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , drop = ['factor_name'] , title = title , show=show and i==0) as (df , fig):
                df = df.transpose()
                ax = fig.add_subplot(111)
                index_mid = np.arange(len(df)) 
                bar_width = 1 / (len(df.columns) + 1)
                for j , bm in enumerate(df.columns):
                    x_pos = index_mid + (j - (len(df.columns) - 1) / 2) * bar_width
                    ax.bar(x_pos  , df[bm], label = bm , width=bar_width)  
                ax.legend(loc='upper right')
                plot.set_xaxis(ax , index_mid , labels = df.index)
                plot.set_yaxis(ax , format='pct' , digits=3 , title = 'Group Excess Return' , title_color='b')
        return group_plot.fig_dict

    def plot_group_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Group CumReturn'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.set_index('date')
                ax = plot.sns_lineplot(df , x='date' , y='cum_ret' , hue='group')

                plot.set_xaxis(ax , df.index.unique() , title = 'Trade Date')
                plot.set_yaxis(ax , format='pct' , digits=2 , title = 'Cummulative Return' , title_color='b')

        return group_plot.fig_dict

    def plot_group_decay(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Groups Return Decay'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.query('stats_name == "decay_grp_ret"')

                ax = plot.sns_barplot(df , x='lag_type' , y='stats_value' , hue='group')
                plot.set_xaxis(ax , title = 'Lag Type')
                plot.set_yaxis(ax , format = 'pct' , digits=2 , title = 'Groups Average Return' , title_color='b')
                
        return group_plot.fig_dict

    def plot_group_ir_decay(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Groups IR Decay'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.query('stats_name == "decay_grp_ir"')
                ax = plot.sns_barplot(df , x='lag_type' , y='stats_value' , hue='group')

                plot.set_xaxis(ax , title = 'Lag Type')
                plot.set_yaxis(ax , format = 'flt' , digits=3 , title = 'Groups Average IR' , title_color='b')

        return group_plot.fig_dict

    def plot_group_year(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Annualized Top Group Performance'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df = df.drop(columns=['abs_avg'])
                plot.plot_table(df.set_index('year') , 
                                int_cols = ['group'] ,
                                pct_cols = ['sum' , 'avg' , 'year_ret', 'std', 'cum_mdd'] , 
                                flt_cols = ['ir'] , pct_ndigit = 3 , stripe_by=1 , emph_last_row=True)
        return group_plot.fig_dict

    def plot_distrib_curve(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Cross-Sectional Distribution'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0 , suptitle=True) as (df , fig):
                    assert not df['date'].duplicated().any() , df['date']
                    lay_out = (3, min(int(np.ceil(len(df) / 3)) , 3))
                    for i in range(min(len(df) , lay_out[0] * lay_out[1])):
                        ax = fig.add_subplot(*lay_out , i + 1)
                        day_df = df.iloc[i]
                        bins = day_df['hist_bins']
                        cnts = day_df['hist_cnts'] / day_df['hist_cnts'].sum()
                        ax.bar(x=bins[:-1] + np.diff(bins) / 2, height=cnts, width=np.diff(bins), color='b' , alpha = 0.5)
                        plot.set_yaxis(ax , format='pct' , digits=0 , tick_size=8 , tick_length=0)
                        plot.set_xaxis(ax , format='flt' , digits=1 , tick_size=8 , tick_length=0 , rotation = None)
                        ax.set_title(str(day_df['date']))
        return group_plot.fig_dict

    def plot_distrib_qtile(self , data : pd.DataFrame , show = False , title_prefix = None):
        group_plot = plot.PlotMultipleData(data , group_key = ['factor_name' , 'benchmark'])
        title = f'{title_prefix or self.title_prefix} Cross-Sectional Quantile'
        for i , sub_data in enumerate(group_plot):     
            with plot.PlotFactorData(sub_data , title = title , show=show and i==0) as (df , fig):
                df.columns.rename('quantile_name', inplace=True)
                df = df.set_index('date').stack().rename('quantile_value').reset_index().set_index('date') # type: ignore
                ax = plot.sns_lineplot(df , x='date' , y='quantile_value' , hue='quantile_name')

                plot.set_xaxis(ax , df.index , title = 'Trade Date')
                plot.set_yaxis(ax , format='flt' , digits=2 , title = 'Quantile Value' , title_color='b')
                
        return group_plot.fig_dict

    """