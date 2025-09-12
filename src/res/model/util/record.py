import pandas as pd
import numpy as np

from src.proj import Logger
from .classes import BaseTrainer

class PredRecorder:
    def __init__(self) -> None:
        self.initialized = False
        
    def initialize(self , trainer : BaseTrainer):
        if self.initialized:
            if self.trainer == trainer: 
                return
            else:
                Logger.critical(f'PredRecorder initialize with {trainer}, but already initialized with another trainer {self.trainer}')
        self.trainer = trainer
        self.preds : dict[str,pd.DataFrame] = {}
        self.dates = trainer.data.test_full_dates
        self.initialized = True

    @property
    def pred_idx(self):
        return f'{self.trainer.model_num}.{self.trainer.model_submodel}.{self.trainer.model_date}.{self.trainer.batch_idx}'

    @property
    def is_empty(self):
        return len(self.preds) == 0
    
    def all_preds(self , interval : int = 1):
        if self.is_empty: 
            return pd.DataFrame()
        elif interval == 1:
            return pd.concat(self.preds.values())
        else:
            dates = self.dates[::interval]
            seq = [pred.loc[np.isin(pred['date'],dates),:] for pred in self.preds.values()]
            return pd.concat(seq)
    
    def append_batch_pred(self):
        if self.pred_idx in self.preds.keys(): 
            return
        if self.trainer.batch_idx < self.trainer.batch_warm_up: 
            return
        
        which_output = self.trainer.model_param.get('which_output' , 0)
        
        secid = self.trainer.data.batch_secid(self.trainer.batch_data)
        date  = self.trainer.data.batch_date(self.trainer.batch_data)

        pred = self.trainer.batch_output.pred_df(secid , date).dropna()
        pred = pred.loc[np.isin(pred['date'],self.dates),:]
        if len(pred) == 0: 
            return

        pred['model_num'] = self.trainer.model_num
        pred['submodel']  = self.trainer.model_submodel
 
        if which_output is None:
            pred['values'] = pred.loc[:,[col for col in pred.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            pred['values'] = pred[f'pred.{which_output}']
        df = pred.loc[:,['model_num' , 'submodel' , 'secid' , 'date' , 'values']]

        self.preds[self.pred_idx] = df

    