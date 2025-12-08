import pandas as pd
import numpy as np
from typing import Any , Literal


from src.proj import Logger
from src import func as FUNC
from src.res.factor.util import StockFactor
from src.res.factor.api import FactorTestAPI , TYPE_of_TASK
from src.res.factor.analytic.test_manager import BaseTestManager
from src.res.model.util import BaseCallBack


class BasicTestResult(BaseCallBack):
    '''basic test result summary'''
    
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        
    @property
    def path_test(self): return self.trainer.path_test_summary
    @property
    def model_test_dates(self) -> np.ndarray: return self.trainer.data.model_test_dates
    def on_test_start(self): 
        self.test_df_date = pd.DataFrame()
        self.test_df_model = pd.DataFrame()

    def on_test_submodel_end(self):
        """update test_df_date and test_df_model"""
        df_date = pd.DataFrame({
            'model_num' : self.status.model_num , 
            'model_date' : self.status.model_date ,
            'submodel' : self.status.model_submodel ,
            'date' : self.model_test_dates ,
            'value' : self.metrics.scores[-len(self.model_test_dates):]
        })
        df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel'])['value'].mean().reset_index()
        df_model['model_date'] = df_model['model_date'].astype(str)
        self.test_df_date = pd.concat([self.test_df_date , df_date])
        self.test_df_model = pd.concat([self.test_df_model , df_model])

    def on_test_end(self): 
        """update test_summary"""
        if self.test_df_model.empty: 
            return
        with Logger.ParagraphIII('Table: Test Summary:'): 
            print(f'Table: Testing Mean Score ({self.config.train_criterion_score}) for Models:')

            df_date = self.test_df_date.copy()
            if df_date['model_date'].nunique() == 1:
                # only one model_date, calculate by year
                df_date['model_date'] = (df_date['date'].astype(int) // 10000).apply(lambda x:f'Y{x}')
            else:
                df_date['model_date'] = df_date['model_date'].astype(str)
                
            df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel'])['value'].mean().reset_index().rename(columns={'model_date':'stat'})

            dfs : dict[str,pd.DataFrame|pd.Series|Any] = {}
            dfs['Avg'] = self.test_df_date.groupby(['model_num','submodel'])['value'].mean()
            dfs['Sum'] = self.test_df_date.groupby(['model_num','submodel'])['value'].sum()
            dfs['Std'] = self.test_df_date.groupby(['model_num','submodel'])['value'].std()

            dfs['T']   = ((dfs['Avg'] / dfs['Std']) * (len(self.test_df_date['date'].unique())**0.5))
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
            
            FUNC.display.display(df_display)
            print(f'Table saved to {self.path_test}')

            # export excel
            rslt = {'summary' : self.status.test_summary , 'by_model' : self.test_df_model}
            for model_num in self.config.model_num_list:
                df : pd.DataFrame = self.test_df_date[self.test_df_date['model_num'] == model_num].pivot_table(
                    'value' , 'date' , 'submodel' , observed=False)
                df_cum = df.cumsum().rename(columns = {submodel:f'{submodel}_cum' for submodel in df.columns})
                df = df.merge(df_cum , on = 'date').rename_axis(None , axis = 'columns')
                rslt[f'{model_num}'] = df
            FUNC.dfs_to_excel(rslt , self.path_test)


class DetailedAlphaAnalysis(BaseCallBack):
    '''factor and portfolio level analysis'''
    CB_ORDER : int = 50
    DISPLAY_TABLES = ['optim@frontface']
    DISPLAY_FIGURES = ['factor@ic_curve@best.market' , 'factor@group_curve@best.market' , 
                       't50@drawdown@best.univ' , 'screen@drawdown@best.univ' , 'revscreen@drawdown@best.univ']

    def __init__(self , trainer , tasks = ['t50' , 'screen'] , # ['factor' , 't50' , 'screen' , 'revscreen'] , 
                 which_model : Literal['avg' , 'first'] = 'avg' , 
                 **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        assert which_model in ['first' , 'avg'] , which_model
        self.which_model = which_model
        assert all(task in FactorTestAPI.TASK_TYPES for task in tasks) , \
            f'ANALYTIC_TASKS must be a list of valid tasks: {FactorTestAPI.TASK_TYPES} , but got {tasks}'
        self.tasks = tasks

    @property
    def path_data(self): return self.trainer.path_analytical_data
    @property
    def path_plot(self): return self.trainer.path_analytical_plot
    @property
    def path_pred(self): return self.trainer.path_pred_dataframe

    def pred_dates(self , interval : int = 1) -> np.ndarray:
        return self.trainer.record.dates[::interval]

    def on_test_end(self):  
        self.test_results : dict[TYPE_of_TASK , BaseTestManager] = {}
        
        if self.trainer.record.is_empty: 
            return
        
        df = self.trainer.record.collect_preds()
        if self.which_model == 'first' or self.trainer.config.Model.n_model == 1:
            df = df.query('model_num == 0')
        else:
            df = df.groupby(['date','secid','submodel'])['pred'].mean().reset_index()
        df = df.set_index(['secid','date'])
        df = df.rename(columns={'submodel':'factor_name'}).pivot_table('pred',['secid','date'],'factor_name').reset_index()
            
        fmp_factor = StockFactor(df.query('date in @self.pred_dates(1)'))
        perf_factor = StockFactor(df.query('date in @self.pred_dates(5)'))
        
        for task in self.tasks:
            with Logger.ParagraphIII(f'{task} test'):
                factor = perf_factor if task == 'factor' else fmp_factor
                self.test_results[task] = FactorTestAPI.run_test(task , factor , title_prefix=self.config.model_name)

        rslts = {f'{task}@{k}':v for task , calc in self.test_results.items() for k,v in calc.get_rslts().items()}
        figs  = {f'{task}@{k}':v for task , calc in self.test_results.items() for k,v in calc.get_figs().items()}

        with Logger.ParagraphIII('Display Analytic Results'):
            self.display_dfs(rslts)
            self.display_figs(figs)

            FUNC.dfs_to_excel(rslts , self.path_data , print_prefix='Analytic datas')
            FUNC.figs_to_pdf(figs , self.path_plot , print_prefix='Analytic plots')

    @classmethod
    def display_dfs(cls , dfs : dict[str , pd.DataFrame]):
        col_pct = ['pf','bm','excess','annualized','mdd','te']
        col_flt = ['ir','calmar','turnover']
        for name in cls.DISPLAY_TABLES:
            if name not in dfs: 
                continue
            df = dfs[name].copy()
            for col in df.columns.intersection(col_pct): 
                df[col] = df[col].map(lambda x:f'{x:.2%}')
            for col in df.columns.intersection(col_flt): 
                df[col] = df[col].map(lambda x:f'{x:.3f}')
            print(f'Table: {name}:')
            FUNC.display.display(df)

    @classmethod
    def display_figs(cls , figs : dict[str , Any]):
        for name in cls.DISPLAY_FIGURES:
            if name not in figs: 
                continue
            print(f'Figure: {name}:')
            FUNC.display.display(figs[name])
        
class GroupReturnAnalysis(BaseCallBack):
    '''group return analysis'''
    CB_ORDER : int = 50
    CB_KEY_PARAMS = ['group_num']
    def __init__(self , trainer , 
                 group_num : int = 20 , **kwargs) -> None:
        self.group_num = group_num
        super().__init__(trainer , **kwargs)

    @property
    def path_grp(self): return str(self.config.model_base_path.rslt('group_return_analysis.xlsx'))

    def on_test_end(self):  
        if self.trainer.record.is_empty: 
            return
        df = self.trainer.record.collect_preds(5)
             
        df['factor_name'] = df['model_num'].astype(str) + '.' + df['submodel']
            
        factor = StockFactor(df.pivot_table('pred',['secid','date'],'factor_name'))
        rslt = {}
        for bm in ['market' , 'csi300' , 'csi500' , 'csi1000']:
            grp = factor.within(bm).eval_group_perf(group_num=self.group_num , excess=True).\
                groupby(['factor_name' , 'group'] , observed=False)['group_ret'].mean().reset_index()
            grp[['model_num', 'submodel']] = grp['factor_name'].str.split('.', expand=True) 
            grp = grp.pivot_table('group_ret',['model_num', 'submodel'],'group' , observed=False).map(lambda x:f'{x:.3%}')
            
            rslt[bm] = grp

        FUNC.dfs_to_excel(rslt , self.path_grp)
        grp : pd.DataFrame = rslt['market']
        grp.index.names = [str(col).replace('model_','') for col in grp.index.names]
        with Logger.ParagraphIII('Table: Grouped Return Results:'):
            FUNC.display.display(grp)
            print(f'Table saved to {self.path_grp}')