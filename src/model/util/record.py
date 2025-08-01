import pandas as pd
from .classes import BaseTrainer

class PredRecorder:
    def __init__(self , every_n_days : int = 1) -> None:
        self.every_n_days = every_n_days
        self.initialized = False

    def initialize(self , trainer : BaseTrainer):
        if self.initialized:
            if self.trainer == trainer: 
                return
            else:
                self.trainer.logger.critical(f'PredRecorder initialize with {trainer}, but already initialized with another trainer {self.trainer}')
        self.trainer = trainer
        self.preds : dict[str,pd.DataFrame] = {}
        self.dates = trainer.data.test_full_dates[::self.every_n_days]
        self.initialized = True

    @property
    def pred_idx(self):
        return f'{self.trainer.model_num}.{self.trainer.model_submodel}.{self.trainer.model_date}.{self.trainer.batch_idx}'

    def append_batch_pred(self):
        if self.pred_idx in self.preds.keys(): return
        if self.trainer.batch_idx < self.trainer.batch_warm_up: return
        
        which_output = self.trainer.model_param.get('which_output' , 0)
        
        secid = self.trainer.data.batch_secid(self.trainer.batch_data)
        date  = self.trainer.data.batch_date(self.trainer.batch_data)

        pred = self.trainer.batch_output.pred_df(secid , date).dropna()
        pred = pred.loc[pred['date'].isin(self.dates),:]
        if len(pred) == 0: return

        pred['model_num'] = self.trainer.model_num
        pred['submodel']  = self.trainer.model_submodel
 
        if which_output is None:
            pred['values'] = pred.loc[:,[col for col in pred.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            pred['values'] = pred[f'pred.{which_output}']
        df = pred.loc[:,['model_num' , 'submodel' , 'secid' , 'date' , 'values']]

        self.preds[self.pred_idx] = df

    @property
    def all_preds(self):
        return pd.concat(list(self.preds.values())) if self.preds else pd.DataFrame()