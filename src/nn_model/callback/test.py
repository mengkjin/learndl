import os
import numpy as np
import pandas as pd

from IPython.display import display
from matplotlib.figure import Figure
from typing import Any

from .base import CallBack
from ...factor.perf.api import PerfManager
from ...factor.fmp.api import FmpManager
from ...func import dfs_to_excel , figs_to_pdf

class DetailedAlphaAnalysis(CallBack):
    '''record and concat each model to Alpha model instance'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module , with_cb=False)

    def init_pred_df(self):
        self.pred_dfs : list[pd.DataFrame] = []
        self.pred_dates = self.data.test_full_dates[::5]

    def append_preds(self):
        if self.batch_idx < self.module.batch_warm_up: return
        
        ij = self.batch_data.i.cpu()
        in_dates = np.isin(self.data.y_date[ij[:,1]] , self.pred_dates)
        if not in_dates.any(): return
        
        full_pred :np.ndarray = self.batch_output.pred.cpu().numpy()[in_dates]
        secid :np.ndarray = self.data.y_secid[ij[:,0]][in_dates]
        date  :np.ndarray = self.data.y_date[ij[:,1]][in_dates]

        assert full_pred.ndim == 2 , full_pred.shape
        if (which_output := self.module.model_param.get('which_output' , 0) ) is None:
            pred = full_pred.mean(axis=-1)
        else:
            pred = full_pred[...,which_output]
        factor_name = f'{self.module.model_num}@{self.module.model_type}'
        df = pd.DataFrame({'factor_name':factor_name,'secid':secid,'date':date,'values':pred})
        self.pred_dfs.append(df)

    def export_pred_df(self): 
        path = f'{self.config.model_base_path}/analysis'
        os.makedirs(path , exist_ok=True)

        df = pd.concat(self.pred_dfs)
        df.set_index(['secid','date']).to_feather(f'{path}/pred_df.feather')
        df = df.pivot_table('values',['secid','date'],'factor_name')
        self.logger.warning(f'Performing Factor and FMP test!')
        fac_man = PerfManager.run_test(df , verbosity = 1)
        fmp_man = FmpManager.run_test(df , verbosity = 1)
        
        rslts : dict[str , pd.DataFrame] = {f'fmp_{k}':v for k,v in fmp_man.get_rslts().items()}
        rslts.update({f'factor_{k}':v for k,v in fac_man.get_rslts().items()})
        figs : dict[str , Figure] = {f'fmp_{k}':v for k,v in fmp_man.get_figs().items()}
        figs.update({f'factor_{k}':v for k,v in fac_man.get_figs().items()})

        print(f'FMP test Result:')
        if 'fmp_prefix' in rslts.keys(): 
            with pd.option_context(
                'display.max_rows', 100,
                'display.max_columns', None,
                'display.width', 1000,
                'display.precision', 3,
                'display.colheader_justify', 'center'):
                df = rslts['fmp_prefix'].copy()
                for col in ['pf','bm','excess','annualized','mdd','te']:  df[col] = df[col].map(lambda x:f'{x:.2%}')
                for col in ['ir','calmar','turnover']:  df[col] = df[col].map(lambda x:f'{x:.3f}')
                display(df)

        dfs_to_excel(rslts , f'{path}_data.xlsx')
        figs_to_pdf(figs , f'{path}_plot.pdf')
        print(f'Analytic datas are saved to {path}_data.xlsx')
        print(f'Analytic plots are saved to {path}_plot.pdf')

    def on_test_start(self):     self.init_pred_df()
    def on_test_batch_end(self): self.append_preds()
    def on_test_end(self):       self.export_pred_df()
