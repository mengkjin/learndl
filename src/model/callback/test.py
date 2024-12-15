import pandas as pd

from typing import Any , Literal

from src import func as FUNC
from src.factor.util import StockFactor
from src.factor.api import FactorTestAPI , TYPE_of_TASK
from src.model.util import BaseCallBack , PredRecorder

PRED_RECORD = PredRecorder()

class DetailedAlphaAnalysis(BaseCallBack):
    ANALYTIC_TASKS : list[TYPE_of_TASK] = ['factor' , 'top'] # 'optim'
    '''record and concat each model to Alpha model instance'''
    def __init__(self , trainer , use_num : Literal['avg' , 'first'] = 'avg' , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.print_info()
        assert use_num in ['first' , 'avg'] , use_num
        self.use_num = use_num
        assert all(task in FactorTestAPI.TASK_TYPES for task in self.ANALYTIC_TASKS) , \
            f'ANALYTIC_TASKS must be a list of valid tasks: {FactorTestAPI.TASK_TYPES} , but got {self.ANALYTIC_TASKS}'

    @property
    def path_data(self): return str(self.config.model_base_path.rslt('data.xlsx'))
    @property
    def path_plot(self): return str(self.config.model_base_path.rslt('plot.pdf'))
    @property
    def path_pred(self): return str(self.config.model_base_path.rslt('pred_df.feather'))

    def on_test_start(self):     PRED_RECORD.initialize(self.trainer)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred(self.trainer)
    def on_test_end(self):       
        if (df := PRED_RECORD.all_preds).empty: return
        if self.use_num == 'first':
            df = df[df['model_num'] == 0]
        else:
            df = df.groupby(['date','secid','submodel'])['values'].mean().reset_index()
        df.set_index(['secid','date']).to_feather(self.path_pred)

        df = df.rename(columns={'submodel':'factor_name'}).pivot_table('values',['secid','date'],'factor_name')
        #self.logger.warning(f'Performing Factor and FMP test!')
        
        self.df = df

        self.test_results = {
            task:FactorTestAPI.run_test(task , StockFactor(df) , verbosity = 1 , write_down=False , display_figs=False)
            for task in self.ANALYTIC_TASKS
        }

        rslts = {f'{task}@{k}':v for task in self.ANALYTIC_TASKS for k,v in self.test_results[task].get_rslts().items()}
        figs  = {f'{task}@{k}':v for task in self.ANALYTIC_TASKS for k,v in self.test_results[task].get_figs().items()}

        if 'optim@frontface' in rslts.keys(): 
            # print(f'FMP test Result:')
            df = rslts['optim@frontface'].copy()
            for col in ['pf','bm','excess','annualized','mdd','te']:  df[col] = df[col].map(lambda x:f'{x:.2%}')
            for col in ['ir','calmar','turnover']:  df[col] = df[col].map(lambda x:f'{x:.3f}')
            FUNC.display.data_frame(df)

        FUNC.dfs_to_excel(rslts , self.path_data , print_prefix='Analytic datas')
        FUNC.figs_to_pdf(figs , self.path_plot , print_prefix='Analytic plots')
        
class GroupReturnAnalysis(BaseCallBack):
    '''record and concat each model to Alpha model instance'''
    def __init__(self , trainer , 
                 group_num : int = 20 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.group_num = group_num

    @property
    def path_grp(self): return str(self.config.model_base_path.rslt('group.xlsx'))

    def on_test_start(self):     PRED_RECORD.initialize(self.trainer)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred(self.trainer)
    def on_test_end(self):       
        if (df := PRED_RECORD.all_preds).empty: return
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
        grp.index.names = [col.replace('model_','') for col in grp.index.names]
        FUNC.display.data_frame(grp , text_after=f'Grouped Return Results are saved to {self.path_grp}')