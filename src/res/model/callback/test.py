import pandas as pd
import numpy as np
from typing import Any , Literal
from matplotlib.figure import Figure

from src.proj import Logger , Timer , Display
from src.basic import DB
from src.func import dfs_to_excel , figs_to_pdf
from src.res.factor.util import StockFactor
from src.res.factor.api import FactorTestAPI
from src.res.model.util import BaseCallBack

class BasicTestResult(BaseCallBack):
    '''basic test result summary'''
    
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        
    @property
    def model_test_dates(self) -> np.ndarray: return self.trainer.data.model_test_dates
    @property
    def test_full_dates(self) -> np.ndarray: return self.trainer.data.test_full_dates
    @property
    def path_summary(self): return self.config.model_base_path.rslt('test_summary.xlsx')
    @property
    def path_test_df(self): return self.config.model_base_path.snapshot('test_by_date.feather')

    def save_test_df(self , vb_level : int = 1):
        df = self.get_test_df()
        DB.save_df(df , self.path_test_df , overwrite = True , vb_level = 99)
        Logger.footnote(f'Basic Test Result saved to {self.path_test_df}' , vb_level = vb_level) 

    def get_test_df(self) -> pd.DataFrame:
        df = pd.concat([DB.load_df(self.path_test_df) , self.test_df_date]) if self.config.is_resuming else self.test_df_date
        df = df.drop_duplicates(subset=['model_num' , 'model_date' , 'submodel' , 'date'] , keep='last').\
                sort_values(by=['model_num' , 'model_date' , 'submodel' , 'date']).reset_index(drop=True)
        return df

    def on_test_start(self): 
        self.test_df_date = pd.DataFrame()

    def on_test_submodel_end(self):
        """update test_df_date and test_df_model"""
        df_date = pd.DataFrame({
            'model_num' : self.status.model_num , 
            'model_date' : self.status.model_date ,
            'submodel' : self.status.model_submodel ,
            'date' : self.model_test_dates ,
            'value' : self.metrics.scores[-len(self.model_test_dates):]
        })
        self.test_df_date = pd.concat([self.test_df_date , df_date])

    def on_test_end(self): 
        """update test_summary"""
        if self.test_df_date.empty: 
            return

        with Logger.ParagraphIII('Test Summary'): 
            self.save_test_df()

            df_date = self.get_test_df().query('date in @self.test_full_dates')
            Logger.caption(f'Table: Test Summary ({self.config.train_criterion_score}) for Models:')

            if df_date['model_date'].nunique() == 1 or self.config.module_type in ['factor' , 'db']:
                # only one model_date, calculate by year
                df_date['model_date'] = (df_date['date'].astype(int) // 10000).apply(lambda x:f'Y{x}')
            else:
                df_date['model_date'] = df_date['model_date'].astype(str)
                
            df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel'])['value'].mean().reset_index().rename(columns={'model_date':'stat'})

            dfs : dict[str,pd.DataFrame|pd.Series|Any] = {}
            dfs['Avg'] = df_date.groupby(['model_num','submodel'])['value'].mean()
            dfs['Sum'] = df_date.groupby(['model_num','submodel'])['value'].sum()
            dfs['Std'] = df_date.groupby(['model_num','submodel'])['value'].std()

            dfs['T']   = ((dfs['Avg'] / dfs['Std']) * (len(df_date['date'].unique())**0.5))
            dfs['IR']  = ((dfs['Avg'] / dfs['Std']) * ((240 / 10)**0.5))

            stat_df = pd.concat([df.reset_index().assign(stat=k) for k,df in dfs.items()])

            # display summary
            df = pd.concat([df_model , stat_df])
            cat_stat = [md for md in df_model['stat'].unique()] + ['Avg' , 'Sum' , 'Std' , 'T' , 'IR']
            cat_subm = ['best' , 'swalast' , 'swabest']

            base_name = self.config.model_module
            if self.config.module_type == 'booster' and self.config.model_booster_optuna: 
                base_name += '.optuna'
            df['model_num'] = df['model_num'].map(lambda x: f'{base_name}.{x}')
            df['submodel']  = pd.Categorical(df['submodel'] , categories = cat_subm, ordered=True) 
            df['stat']      = pd.Categorical(df['stat']     , categories = cat_stat, ordered=True) 

            self.status.test_summary = df.rename(columns={'model_num':'model'}).pivot_table('value' , 'stat' , ['model' , 'submodel'] , observed=False).round(4)

            # more than 100 rows of test_df_model means the cycle is month / day
            df_display = self.status.test_summary
            if len(df_display) > 100: 
                df_display = df_display.loc[['Avg' , 'Sum' , 'Std' , 'T' , 'IR']]           
            Display(df_display)
            
            # export excel
            rslt = {'test_summary' : self.status.test_summary , 'test_by_model' : df_model}
            for model_num in self.config.model_num_list:
                df : pd.DataFrame = df_date[df_date['model_num'] == model_num].pivot_table(
                    'value' , 'date' , 'submodel' , observed=False)
                df_cum = df.cumsum().rename(columns = {submodel:f'{submodel}_cum' for submodel in df.columns})
                df = df.merge(df_cum , on = 'date').rename_axis(None , axis = 'columns')
                rslt[f'{model_num}'] = df
            dfs_to_excel(rslt , self.path_summary , print_prefix = 'Test Summary')


class DetailedAlphaAnalysis(BaseCallBack):
    '''factor and portfolio level analysis'''
    CB_ORDER : int = 50
    CB_KEY_PARAMS = ['tasks']
    DISPLAY_TABLES = ['factor@frontface']
    DISPLAY_FIGURES = ['factor@ic_curve@best.market' , 'factor@group_return@best' ,
                       't50@drawdown@best.univ' , 'screen@drawdown@best.univ' , 'revscreen@drawdown@best.univ']

    def __init__(self , trainer , tasks = ['factor' , 't50' , 'screen' , 'revscreen'] , **kwargs) -> None:
        assert all(task in FactorTestAPI.TEST_TYPES for task in tasks) , \
            f'TASKS must be a list of valid tasks: {FactorTestAPI.TEST_TYPES} , but got {tasks}'

        self.tasks = ','.join(tasks)
        self.factor_tasks = [task for task in tasks if task in ['factor']]
        self.fmp_tasks = [task for task in tasks if task not in ['factor']]
        super().__init__(trainer , **kwargs)

        self.test_results : dict[str , pd.DataFrame] = {}
        self.test_figures : dict[str , Figure] = {}
        
    @property
    def test_path(self): return self.config.model_base_path.snapshot()
    @property
    def path_data(self): return self.trainer.path_analytical_data
    @property
    def path_plot(self): return self.trainer.path_analytical_plot
    @property
    def display_tables(self) -> list[str]: return [name for name in self.DISPLAY_TABLES if name in self.test_results]
    @property
    def display_figures(self) -> list[str]: return [name for name in self.DISPLAY_FIGURES if name in self.test_figures]
    @property
    def factor_names(self) -> list[str] | Any: 
        return self.trainer.model_submodels

    def get_factor(self , tested_only = True , interval = 5 , which : Literal['first' , 'avg'] = 'avg') -> StockFactor:
        df = self.trainer.record.get_preds(tested_only=tested_only , interval = interval)
        if which == 'first' or self.trainer.config.Model.n_model == 1:
            df['factor_name'] = df['submodel']
            df = df.query('model_num == 0')
        elif which == 'avg':
            df['factor_name'] = df['submodel']
            df = df.groupby(['date','secid','submodel'])['pred'].mean().reset_index()
        else:
            raise ValueError(f'Invalid which: {which}')
        df = df.pivot_table('pred',['secid','date'],'factor_name').reset_index()
        factor = StockFactor(df , factor_names = self.factor_names)
        return factor

    def factor_test(self , indent : int = 0 , vb_level : int = 1):
        with Logger.ParagraphIII('Factor Perf Test'):
            with Timer(f'FactorPerfTest.get_factor' , indent = indent , vb_level = vb_level):
                factor = self.get_factor(tested_only=False , interval = 5)
            with Timer(f'FactorPerfTest.load_day_rets' , indent = indent , vb_level = vb_level):
                factor.day_returns()

            for task in self.factor_tasks:
                Logger.divider(vb_level = vb_level)
                results = FactorTestAPI.run_test(task , factor , test_path = self.test_path , 
                                                 resume = self.config.is_resuming , save_resumable = True , 
                                                 start_dt = self.trainer.config.beg_date , end_dt = self.trainer.config.end_date ,
                                                 indent = indent , vb_level = vb_level,
                                                 title_prefix=self.config.model_name)

                self.test_results.update({f'{task}@{k}':v for k,v in results.get_rslts().items()})
                self.test_figures.update({f'{task}@{k}':v for k,v in results.get_figs().items()})

    def fmp_test(self , indent : int = 0 , vb_level : int = 1):
        with Logger.ParagraphIII('Factor FMP Test'):
            with Timer(f'FactorFMPTest.get_factor' , indent = indent , vb_level = vb_level):
                factor = self.get_factor(tested_only=True , interval = 1)
            with Timer(f'FactorFMPTest.load_alpha_models' , indent = indent , vb_level = vb_level):
                factor.alpha_models()
            with Timer(f'FactorFMPTest.load_risk_models' , indent = indent , vb_level = vb_level):
                factor.risk_model()
            with Timer(f'FactorFMPTest.load_universe' , indent = indent , vb_level = vb_level):
                factor.universe(load = True)
            with Timer(f'FactorFMPTest.load_day_quotes' , indent = indent , vb_level = vb_level):
                factor.day_quotes()
            for task in self.fmp_tasks:
                Logger.divider(vb_level = vb_level)
                results = FactorTestAPI.run_test(task , factor , test_path = self.test_path , 
                                                 resume = self.config.is_resuming , save_resumable = True , 
                                                 start_dt = self.trainer.config.beg_date , end_dt = self.trainer.config.end_date,
                                                 indent = indent , vb_level = vb_level,
                                                 title_prefix=self.config.model_name)

                self.test_results.update({f'{task}@{k}':v for k,v in results.get_rslts().items()})
                self.test_figures.update({f'{task}@{k}':v for k,v in results.get_figs().items()})

    def display_export(self):
        with Logger.ParagraphIII('Display Analytic Results'):
            for name in self.display_tables:
                Logger.caption(f'Table: {name.title()}:')
                df = self.test_results[name].copy()
                df = df.reset_index(drop=isinstance(df.index , pd.RangeIndex))
                for col in df.columns:
                    if col in ['pf','bm','excess','annualized','mdd','te','ret']: 
                        df[col] = df[col].map(lambda x:f'{x:.2%}')
                    elif col in ['ir','calmar','turnover','IC_avg' , 'IC_std' , 'IC(ann)' , 'ICIR', 'IC_mdd' , '|IC|_avg']: 
                        df[col] = df[col].map(lambda x:f'{x:.3f}')
                    elif df.columns.name in ['group'] and (isinstance(col , int) or str(col).isdigit()):
                        df[col] = df[col].map(lambda x:f'{x:.3%}')
                Display(df)

            for name in self.display_figures:
                Logger.caption(f'Figure: {name.title()}:')
                Display(self.test_figures[name])

            dfs_to_excel(self.test_results , self.path_data , print_prefix='Analytic datas')
            figs_to_pdf(self.test_figures , self.path_plot , print_prefix='Analytic plots')

    def on_test_end(self):
        if not self.tasks:
            return
        self.factor_test()
        self.fmp_test()
        self.display_export()