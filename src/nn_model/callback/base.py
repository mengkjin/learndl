import logging
import numpy as np
import pandas as pd

from IPython.display import display 
from ..classes import BaseCB , TrainerStatus , BaseDataModule , BaseTrainer , BatchData , BoosterData
from ..util import Checkpoint , Metrics , TrainConfig

class CallBack(BaseCB):
    def __init__(self , model_module : BaseTrainer , with_cb : bool , print_info = True , turn_off = False , *args , **kwargs) -> None:
        super().__init__(model_module , with_cb , turn_off)
        if print_info: self._print_info(depth=1)
    @property
    def config(self) -> TrainConfig:    return self.module.config
    @property
    def logger(self) -> logging.Logger: return self.module.logger
    @property
    def status(self) -> TrainerStatus:  return self.module.status
    @property
    def metrics(self) -> Metrics :  return self.module.metrics
    @property
    def ckpt(self) -> Checkpoint: return self.module.checkpoint
    @property
    def data(self) -> BaseDataModule: return self.module.data
    @property
    def batch_data(self): return self.module.batch_data
    @property
    def batch_idx(self): return self.module.batch_idx
    @property
    def batch_output(self): return self.module.batch_output

def df_display(df):
    with pd.option_context(
        'display.max_rows', 100,
        'display.max_columns', None,
        'display.width', 1000,
        'display.precision', 3,
        'display.colheader_justify', 'center'):
        display(df)

class PredRecorder:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self , module : BaseTrainer):
        if not self.initialized:
            self.preds : dict[str,pd.DataFrame] = {}
            self.dates = module.data.test_full_dates[::5]
            self.initialized = True

    def append_batch_pred(self , module : BaseTrainer):
        pred_idx = f'{module.model_num}.{module.model_type}.{module.batch_idx}'
        if pred_idx in self.preds.keys(): return
        if module.batch_idx < module.batch_warm_up: return
        
        which_output = module.model_param.get('which_output' , 0)
        if isinstance(module.batch_data , BatchData):
            ij = module.batch_data.i.cpu()
            secid , date = module.data.y_secid[ij[:,0]] , module.data.y_date[ij[:,1]]
        elif isinstance(module.batch_data , BoosterData):
            secid , date = module.batch_data.SECID() , module.batch_data.DATE()
        pred = module.batch_output.pred_df(secid , date)
        pred = pred.loc[pred['date'].isin(self.dates),:]
        if len(pred) == 0: return

        pred['model_num'] = module.model_num
        pred['model_type'] = module.model_type
 
        if which_output is None:
            pred['values'] = pred.loc[:,[col for col in pred.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            pred['values'] = pred[f'pred.{which_output}']
        df = pred.loc[:,['model_num' , 'model_type' , 'secid' , 'date' , 'values']]
        self.preds[pred_idx] = df

    @property
    def all_preds(self):
        return pd.concat(list(self.preds.values()))