import pandas as pd
import numpy as np
from typing import Any , Literal


from src.proj import Logger , Timer
from src.basic import DB
from src import func as FUNC
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
    def path_summary(self): return self.config.model_base_path.rslt('test_summary.xlsx')
    @property
    def path_test_df(self): return self.config.model_base_path.rslt('test_by_date.feather')

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
        df_model = df_date.groupby(['model_num' , 'model_date' , 'submodel'])['value'].mean().reset_index()
        df_model['model_date'] = df_model['model_date'].astype(str)
        self.test_df_date = pd.concat([self.test_df_date , df_date])

    def on_test_end(self): 
        """update test_summary"""
        df_date = pd.concat([DB.load_df(self.path_test_df) , self.test_df_date])
        if df_date.empty: 
            return

        df_date = df_date.drop_duplicates(subset=['model_num' , 'model_date' , 'submodel' , 'date'] , keep='last').\
            sort_values(by=['model_num' , 'model_date' , 'submodel' , 'date'])

        DB.save_df(df_date , self.path_test_df , overwrite = True , verbose = False)
        print(f'Basic Test Result saved to {self.path_test_df}')
        
        with Logger.ParagraphIII('Table: Test Summary:'): 
            print(f'Table: Testing Mean Score ({self.config.train_criterion_score}) for Models:')

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
            
            FUNC.display.display(df_display)
            

            # export excel
            rslt = {'test_summary' : self.status.test_summary , 'test_by_model' : df_model}
            for model_num in self.config.model_num_list:
                df : pd.DataFrame = df_date[df_date['model_num'] == model_num].pivot_table(
                    'value' , 'date' , 'submodel' , observed=False)
                df_cum = df.cumsum().rename(columns = {submodel:f'{submodel}_cum' for submodel in df.columns})
                df = df.merge(df_cum , on = 'date').rename_axis(None , axis = 'columns')
                rslt[f'{model_num}'] = df
            FUNC.dfs_to_excel(rslt , self.path_summary)
            print(f'Table saved to {self.path_summary}')


class DetailedAlphaAnalysis(BaseCallBack):
    '''factor and portfolio level analysis'''
    CB_ORDER : int = 50
    CB_KEY_PARAMS = ['tasks']
    DISPLAY_TABLES = ['optim@frontface' , 'group@market']
    DISPLAY_FIGURES = ['factor@ic_curve@best.market' , 'factor@group_curve@best.market' , 
                       't50@drawdown@best.univ' , 'screen@drawdown@best.univ' , 'revscreen@drawdown@best.univ']

    def __init__(self , trainer , tasks = ['factor' , 't50' , 'screen' , 'revscreen' , 'group'] , **kwargs) -> None:
        assert all(task in FactorTestAPI.TASK_TYPES + ['group'] for task in tasks) , \
            f'TASKS must be a list of valid tasks: {FactorTestAPI.TASK_TYPES + ['group']} , but got {tasks}'

        self.tasks = tasks
        self.group_task = 'group' in tasks
        self.factor_task = 'factor' in tasks
        self.fmp_tasks = [task for task in tasks if task not in ['group' , 'factor']]
        super().__init__(trainer , **kwargs)
        
        
    @property
    def test_path(self): return self.config.model_base_path.rslt()
    @property
    def path_data(self): return self.trainer.path_analytical_data
    @property
    def path_plot(self): return self.trainer.path_analytical_plot

    def pred_to_factor(self , df : pd.DataFrame , which : Literal['first' , 'avg' , 'all'] = 'avg') -> StockFactor:
        if which == 'all':
            df['factor_name'] = df['model_num'].astype(str) + '.' + df['submodel']
        elif which == 'first' or self.trainer.config.Model.n_model == 1:
            df['factor_name'] = df['submodel']
            df = df.query('model_num == 0')
        elif which == 'avg':
            df['factor_name'] = df['submodel']
            df = df.groupby(['date','secid','submodel'])['pred'].mean().reset_index()
        else:
            raise ValueError(f'Invalid which: {which}')
        df = df.pivot_table('pred',['secid','date'],'factor_name').reset_index()
        return StockFactor(df)

    def get_fmp_factor(self) -> StockFactor:
        df = self.trainer.record.get_preds(include_resume=False , interval = 1)
        return self.pred_to_factor(df , 'avg')

    def get_perf_factor(self) -> StockFactor:
        df = self.trainer.record.get_preds(include_resume=True , interval = 5)
        return self.pred_to_factor(df , 'avg')

    def get_group_factor(self) -> StockFactor:
        df = self.trainer.record.get_preds(include_resume=True , interval = 5)
        return self.pred_to_factor(df , 'all')

    def fmp_test(self):
        factor = None
        for task in self.fmp_tasks:
            with Logger.ParagraphIII(f'{task} test'):
                with Timer(f'{task.title()}FMPTest.get_factor'):
                    factor = self.get_fmp_factor() if factor is None else factor
                results = FactorTestAPI.run_test(task , factor , test_path = self.test_path , 
                                                 resume = self.config.is_resuming , save_resumable = True , 
                                                 title_prefix=self.config.model_name)

                self.test_results.update({f'{task}@{k}':v for k,v in results.get_rslts().items()})
                self.test_figures.update({f'{task}@{k}':v for k,v in results.get_figs().items()})


    def perf_test(self):
        with Logger.ParagraphIII(f'factor perf test'):
            with Timer(f'FactorPerfTest.get_factor'):
                factor = self.get_perf_factor()
            result = FactorTestAPI.run_test('factor' , factor , test_path = self.test_path , 
                                            resume = self.config.is_resuming , save_resumable = True , 
                                            title_prefix=self.config.model_name)

            self.test_results.update({f'factor@{k}':v for k,v in result.get_rslts().items()})
            self.test_figures.update({f'factor@{k}':v for k,v in result.get_figs().items()})
                          

    def group_test(self , group_num : int = 20 , benchmarks = ['market' , 'csi300' , 'csi500' , 'csi1000']):
        with Logger.ParagraphIII(f'group return test'):
            with Timer(f'GroupReturnTest.get_factor'):
                factor = self.get_group_factor()
            with Timer(f'GroupReturnTest.calc'):
                results : dict[str , pd.DataFrame] = {}
                for bm in benchmarks:
                    grp = factor.within(bm).eval_group_perf(group_num=group_num , excess=True).\
                        groupby(['factor_name' , 'group'] , observed=False)['group_ret'].mean().reset_index()
                    grp[['model_num', 'submodel']] = grp['factor_name'].str.split('.', expand=True) 
                    grp = grp.pivot_table('group_ret',['model_num', 'submodel'],'group' , observed=False).map(lambda x:f'{x:.3%}')
                    grp.index.names = [str(col).replace('model_','') for col in grp.index.names]
                    results[bm] = grp

            self.test_results.update({f'group@{k}':v for k,v in results.items()})


    def on_test_end(self):
        self.test_results = {}
        self.test_figures = {}
        
        self.group_test()
        self.perf_test()
        self.fmp_test()

        with Logger.ParagraphIII('Display Analytic Results'):
            self.display_dfs(self.test_results)
            self.display_figs(self.test_figures)

            FUNC.dfs_to_excel(self.test_results , self.path_data , print_prefix='Analytic datas')
            FUNC.figs_to_pdf(self.test_figures , self.path_plot , print_prefix='Analytic plots')


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