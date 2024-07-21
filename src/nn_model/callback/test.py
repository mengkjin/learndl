import os
import numpy as np
import pandas as pd

from IPython.display import display
from matplotlib.figure import Figure
from typing import Any , Literal

from .base import CallBack , df_display , PredRecorder
from ...factor.perf.api import PerfManager
from ...factor.fmp.api import FmpManager
from ...factor.perf.stat import calc_grp_perf
from ...func import dfs_to_excel , figs_to_pdf

PRED_RECORD = PredRecorder()

class DetailedAlphaAnalysis(CallBack):
    '''record and concat each model to Alpha model instance'''
    def __init__(self , model_module , 
                 use_num : Literal['avg' , 'first'] = 'avg' , **kwargs) -> None:
        super().__init__(model_module , with_cb=False , **kwargs)
        assert use_num in ['first' , 'avg'] , use_num
        self._use_num = use_num

    def on_test_start(self):     PRED_RECORD.initialize(self.module)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred(self.module)
    def on_test_end(self):       
        path = f'{self.config.model_base_path}/detailed_analysis'
        os.makedirs(path , exist_ok=True)

        df = PRED_RECORD.all_preds
        if self._use_num == 'first':
            df = df[df['model_num'] == 0]
        else:
            df = df.groupby(['date','secid','model_type'])['values'].mean().reset_index()
        df.set_index(['secid','date']).to_feather(f'{path}/pred_df.feather')
        df = df.rename(columns={'model_type':'factor_name'}).pivot_table('values',['secid','date'],'factor_name')
        self.logger.warning(f'Performing Factor and FMP test!')
        self.fac_man = PerfManager.run_test(df , verbosity = 1)
        self.fmp_man = FmpManager.run_test(df , verbosity = 1)
        
        rslts : dict[str , pd.DataFrame] = {f'fmp_{k}':v for k,v in self.fmp_man.get_rslts().items()}
        rslts.update({f'factor_{k}':v for k,v in self.fac_man.get_rslts().items()})
        figs : dict[str , Figure] = {f'fmp_{k}':v for k,v in self.fmp_man.get_figs().items()}
        figs.update({f'factor_{k}':v for k,v in self.fac_man.get_figs().items()})

        if 'fmp_prefix' in rslts.keys(): 
            print(f'FMP test Result:')
            df = rslts['fmp_prefix'].copy()
            for col in ['pf','bm','excess','annualized','mdd','te']:  df[col] = df[col].map(lambda x:f'{x:.2%}')
            for col in ['ir','calmar','turnover']:  df[col] = df[col].map(lambda x:f'{x:.3f}')
            df_display(df)

        dfs_to_excel(rslts , f'{path}/data.xlsx')
        figs_to_pdf(figs , f'{path}/plot.pdf')
        print(f'Analytic datas are saved to {path}/data.xlsx')
        print(f'Analytic plots are saved to {path}/plot.pdf')

class GroupReturnAnalysis(CallBack):
    '''record and concat each model to Alpha model instance'''
    def __init__(self , model_module , 
                 group_num : int = 20 , **kwargs) -> None:
        super().__init__(model_module , with_cb=False , **kwargs)
        self._group_num = group_num

    def on_test_start(self):     PRED_RECORD.initialize(self.module)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred(self.module)
    def on_test_end(self):       
        df = PRED_RECORD.all_preds
        df['factor_name'] = df['model_num'].astype(str) + '.' + df['model_type']
        df = df.pivot_table('values',['secid','date'],'factor_name')
        df = calc_grp_perf(df , group_num=self._group_num)

        df = df.groupby(['factor_name' , 'group'] , observed=False)['group_ret'].mean().reset_index()
        df[['model_num', 'model_type']] = df['factor_name'].str.split('.', expand=True) 
        df = df.pivot_table('group_ret','group',['model_num', 'model_type'], observed=False).map(lambda x:f'{x:.3%}')
        
        print(f'Grouped Return Result:')
        df_display(df)