import os
import pandas as pd

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
        self.pred_df : pd.DataFrame | Any = None

    def append_preds(self):
        if self.batch_idx < self.module.batch_warm_up: return
        which_output = self.module.model_param.get('which_output' , 0)
        full_pred = self.batch_output.pred.cpu()
        i0 = self.batch_data.i[:,0].cpu()
        i1 = self.batch_data.i[:,1].cpu()
        secid = self.data.y_secid[i0]
        date  = self.data.y_date[i1]
        assert full_pred.ndim == 2 , full_pred.shape
        if which_output is None:
            pred = full_pred.mean(dim=-1)
        else:
            pred = full_pred[...,which_output]
        factor_name = f'{self.module.model_num}@{self.module.model_type}'
        df = pd.DataFrame({'factor_name':factor_name,'secid':secid,'date':date,'values':pred})
        if self.pred_df is None:
            self.pred_df = df
        elif isinstance(self.pred_df , pd.DataFrame):
            self.pred_df = pd.concat([self.pred_df , df])
        else:
            assert TypeError(self.pred_df)

    def export_pred_df(self): 
        path = f'{self.config.model_base_path}/analysis'
        os.makedirs(path , exist_ok=True)

        dates = self.pred_df['date'].unique()[::5]
        self.pred_df.set_index(['secid','date']).to_feather(f'{path}/pred_df.feather')
        df = self.pred_df[self.pred_df['date'].isin(dates)].pivot_table('values',['secid','date'],'factor_name')
        fac_man = PerfManager.run_test(df)
        fmp_man = FmpManager.run_test(df , verbosity = 1)
        
        rslts : dict[str , pd.DataFrame] = {f'fmp_{k}':v for k,v in fmp_man.get_rslts().items()}
        rslts.update({f'factor_{k}':v for k,v in fac_man.get_rslts().items()})
        figs : dict[str , Figure] = {f'fmp_{k}':v for k,v in fmp_man.get_figs().items()}
        figs.update({f'factor_{k}':v for k,v in fac_man.get_figs().items()})

        if 'fmp_prefix' in rslts.keys(): print(rslts['fmp_prefix'])
        dfs_to_excel(rslts , f'{path}_data.xlsx')
        figs_to_pdf(figs , f'{path}_plot.pdf')
        print(f'Analytic datas are saved to {path}_data.xlsx')
        print(f'Analytic plots are saved to {path}_plot.pdf')

    def on_test_start(self):     self.init_pred_df()
    def on_test_batch_end(self): self.append_preds()
    def on_test_end(self):       self.export_pred_df()
