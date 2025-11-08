import pandas as pd
from typing import Any , Literal

from src.proj import Logger
from src import func as FUNC
from src.basic import DB
from src.res.factor.util import StockFactor
from src.res.factor.api import FactorTestAPI , TYPE_of_TASK
from src.res.factor.analytic.test_manager import BaseTestManager
from src.res.model.util import BaseCallBack , PredRecorder

PRED_RECORD = PredRecorder()
TEST_TYPES : list[TYPE_of_TASK] = ['factor' , 't50' , 'screen']

class DetailedAlphaAnalysis(BaseCallBack):
    DISPLAY_TABLES = ['optim@frontface']
    DISPLAY_FIGURES = ['factor@ic_curve@best.market' , 'factor@group_curve@best.market' , 't50@perf_drawdown@best.univ' , 'screen@perf_drawdown@best.univ']
    '''record and concat each model to Alpha model instance'''
    def __init__(self , trainer , use_num : Literal['avg' , 'first'] = 'avg' , 
                 tasks = TEST_TYPES , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.print_info()
        assert use_num in ['first' , 'avg'] , use_num
        self.use_num = use_num
        assert all(task in FactorTestAPI.TASK_TYPES for task in tasks) , \
            f'ANALYTIC_TASKS must be a list of valid tasks: {FactorTestAPI.TASK_TYPES} , but got {tasks}'
        self.tasks = tasks
        if 'screen' not in self.tasks:
            self.tasks += 'screen'

    @property
    def analytic_tasks(self) -> list[TYPE_of_TASK]:
        tasks = [*self.tasks]
        for task in TEST_TYPES:
            if task not in self.tasks:
                tasks += task
        return tasks

    @property
    def path_data(self): return self.trainer.path_analytical_data
    @property
    def path_plot(self): return self.trainer.path_analytical_plot
    @property
    def path_pred(self): return self.trainer.path_pred_dataframe

    def on_test_start(self):     PRED_RECORD.initialize(self.trainer)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred()
    def on_test_end(self):  
        if PRED_RECORD.is_empty: 
            return
        df = PRED_RECORD.all_preds()
        DB.save_df(df , self.path_pred , overwrite = True)
        if self.use_num == 'first':
            df = df.query('model_num == 0')
        else:
            df = df.groupby(['date','secid','submodel'])['values'].mean().reset_index()
        df = df.set_index(['secid','date'])
        df = df.rename(columns={'submodel':'factor_name'}).pivot_table('values',['secid','date'],'factor_name')
            
        factors : dict[int , StockFactor] = {}
        self.test_results : dict[TYPE_of_TASK , BaseTestManager] = {}
        for task in self.analytic_tasks:
            with Logger.EnclosedMessage(f'{task} test'):
                interval = 1 if task in ['t50' , 'screen'] else 5
                if interval not in factors.keys():
                    dates = PRED_RECORD.dates[::interval] # noqa
                    factors[interval] = StockFactor(df.reset_index().query('date in @dates').set_index(['secid','date']))
                factor = factors[interval]
                self.test_results[task] = FactorTestAPI.run_test(
                    task , factor , verbosity = 1 , write_down=False , display_figs=False , title_prefix=self.config.model_name)

        rslts = {f'{task}@{k}':v for task , calc in self.test_results.items() for k,v in calc.get_rslts().items()}
        figs  = {f'{task}@{k}':v for task , calc in self.test_results.items() for k,v in calc.get_figs().items()}

        with Logger.EnclosedMessage('Display Analytic Results' , timer = False):
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
    '''record and concat each model to Alpha model instance'''
    def __init__(self , trainer , 
                 group_num : int = 20 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.group_num = group_num

    @property
    def path_grp(self): return str(self.config.model_base_path.rslt('group.xlsx'))

    def on_test_start(self):     PRED_RECORD.initialize(self.trainer)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred()
    def on_test_end(self):  
        if PRED_RECORD.is_empty: 
            return
        df = PRED_RECORD.all_preds(5)
             
        df['factor_name'] = df['model_num'].astype(str) + '.' + df['submodel']
            
        factor = StockFactor(df.pivot_table('values',['secid','date'],'factor_name'))
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
        with Logger.EnclosedMessage('Table: Grouped Return Results:' , timer = False):
            FUNC.display.display(grp)
            print(f'Table saved to {self.path_grp}')