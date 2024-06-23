import pandas as pd

from typing import Any

from .base import CallBack
from ...factor.perf.api import PerfManager
from ...factor.fmp.api import FmpManager

class ExportPredDF(CallBack):
    '''record and concat each model to Alpha model instance'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module , with_cb=False)

    def init_pred_df(self):
        self.pred_df : pd.DataFrame | Any = None

    def append_preds(self):
        if self.module.batch_idx < self.module.batch_warm_up: return

        pred = self.module.batch_output.pred.cpu()
        i0 = self.module.batch_data.i[:,0].cpu()
        i1 = self.module.batch_data.i[:,1].cpu()
        secid = self.data.y_secid[i0]
        date  = self.data.y_date[i1]
        assert pred.ndim == 2 , pred.shape

        dfs = [pd.DataFrame({'factor_name':f'{self.module.model_num}_{self.module.model_type}_{i}',
                             'secid':secid,'date':date,'values':pred[...,i].flatten()}) for i in range(pred.shape[-1])]

        if self.pred_df is None:
            self.pred_df = pd.concat(dfs) 
        elif isinstance(self.pred_df , pd.DataFrame):
            self.pred_df = pd.concat([self.pred_df , *dfs])
        else:
            assert TypeError(self.pred_df)

    def export_pred_df(self): 
        path = f'{self.config.model_base_path}/pred_analysis'

        self.pred_df = self.pred_df.set_index(['secid','date'])
        self.pred_df.to_feather(f'{path}/pred_df.feather')
        
        df = self.pred_df.pivot_table('values',['secid','date'],'factor_name')
        perf_man = PerfManager.run_test(df).save(path)
        fmp_man = FmpManager.run_test(df , verbosity = 2).save(path)
        if 'prefix' in fmp_man.perf_calc_dict.keys():
            print(fmp_man.perf_calc_dict['prefix'].calc_rslt)
        self.pred_df.to_feather(f'{path}/pred_df.csv')

    def on_test_model_start(self):   self.init_pred_df()
    def on_test_batch_end(self):     self.append_preds()
    def on_test_model_end(self):     self.export_pred_df()
