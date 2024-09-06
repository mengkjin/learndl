import pandas as pd

from matplotlib.figure import Figure
from typing import Any , Literal

from .util import display
from .util.record import PredRecorder
from ..util.classes import BaseCallBack
from ...factor.perf.api import PerfManager
from ...factor.fmp.api import FmpManager
from ...factor.perf.stat import calc_grp_perf
from ...func import dfs_to_excel , figs_to_pdf

PRED_RECORD = PredRecorder()

class DetailedAlphaAnalysis(BaseCallBack):
    '''record and concat each model to Alpha model instance'''
    def __init__(self , trainer , use_num : Literal['avg' , 'first'] = 'avg' , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.print_info()
        assert use_num in ['first' , 'avg'] , use_num
        self.use_num = use_num

    @property
    def path_data(self): return str(self.config.model_base_path.rslt('data.xlsx'))
    @property
    def path_plot(self): return str(self.config.model_base_path.rslt('plot.pdf'))
    @property
    def path_pred(self): return str(self.config.model_base_path.rslt('pred_df.feather'))

    def on_test_start(self):     PRED_RECORD.initialize(self.trainer)
    def on_test_batch_end(self): PRED_RECORD.append_batch_pred(self.trainer)
    def on_test_end(self):       

        df = PRED_RECORD.all_preds
        if self.use_num == 'first':
            df = df[df['model_num'] == 0]
        else:
            df = df.groupby(['date','secid','model_type'])['values'].mean().reset_index()
        df.set_index(['secid','date']).to_feather(self.path_pred)

        df = df.rename(columns={'model_type':'factor_name'}).pivot_table('values',['secid','date'],'factor_name')
        #self.logger.warning(f'Performing Factor and FMP test!')
        
        self.df = df
        self.fac_man = PerfManager.run_test(df , verbosity = 0)
        self.fmp_man = FmpManager.run_test(df , verbosity = 0)
        
        rslts : dict[str , pd.DataFrame] = {f'fmp_{k}':v for k,v in self.fmp_man.get_rslts().items()}
        rslts.update({f'factor_{k}':v for k,v in self.fac_man.get_rslts().items()})
        figs : dict[str , Figure] = {f'fmp_{k}':v for k,v in self.fmp_man.get_figs().items()}
        figs.update({f'factor_{k}':v for k,v in self.fac_man.get_figs().items()})

        if 'fmp_prefix' in rslts.keys(): 
            # print(f'FMP test Result:')
            df = rslts['fmp_prefix'].copy()
            for col in ['pf','bm','excess','annualized','mdd','te']:  df[col] = df[col].map(lambda x:f'{x:.2%}')
            for col in ['ir','calmar','turnover']:  df[col] = df[col].map(lambda x:f'{x:.3f}')
            display.data_frame(df)

        dfs_to_excel(rslts , self.path_data)
        figs_to_pdf(figs , self.path_plot)
        
        print(f'Analytic datas are saved to {self.path_data}')
        print(f'Analytic plots are saved to {self.path_plot}')

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
        df = PRED_RECORD.all_preds
        df['factor_name'] = df['model_num'].astype(str) + '.' + df['model_type']
        df = df.pivot_table('values',['secid','date'],'factor_name')

        rslt = {}
        for bm in ['market' , 'csi300' , 'csi500' , 'csi1000']:
            benchmark = None if bm == 'market' else bm
            grp = calc_grp_perf(df , benchmark = benchmark , group_num=self.group_num , excess=True)

            grp = grp.groupby(['factor_name' , 'group'] , observed=False)['group_ret'].mean().reset_index()
            grp[['model_num', 'model_type']] = grp['factor_name'].str.split('.', expand=True) 
            grp = grp.pivot_table('group_ret',['model_num', 'model_type'],'group' , observed=False).map(lambda x:f'{x:.3%}')
            
            rslt[bm] = grp

        dfs_to_excel(rslt , self.path_grp)
        grp : pd.DataFrame = rslt['market']
        grp.index.names = [col.replace('model_','') for col in grp.index.names]
        display.data_frame(grp)
        print(f'Grouped Return Results are saved to {self.path_grp}')