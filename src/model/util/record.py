import pandas as pd

from .classes import BaseTrainer

class PredRecorder:
    def __init__(self) -> None:
        ...

    def initialize(self , trainer : BaseTrainer):
        self.preds : dict[str,pd.DataFrame] = {}
        self.dates = trainer.data.test_full_dates[::5]
        self.initialized = True

    def append_batch_pred(self , trainer : BaseTrainer):
        pred_idx = f'{trainer.model_num}.{trainer.model_submodel}.{trainer.model_date}.{trainer.batch_idx}'
        if pred_idx in self.preds.keys(): return
        if trainer.batch_idx < trainer.batch_warm_up: return
        
        which_output = trainer.model_param.get('which_output' , 0)
        
        secid = trainer.data.batch_secid(trainer.batch_data)
        date  = trainer.data.batch_date(trainer.batch_data)

        pred = trainer.batch_output.pred_df(secid , date).dropna()
        pred = pred.loc[pred['date'].isin(self.dates),:]
        if len(pred) == 0: return

        pred['model_num'] = trainer.model_num
        pred['submodel']  = trainer.model_submodel
 
        if which_output is None:
            pred['values'] = pred.loc[:,[col for col in pred.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            pred['values'] = pred[f'pred.{which_output}']
        df = pred.loc[:,['model_num' , 'submodel' , 'secid' , 'date' , 'values']]

        self.preds[pred_idx] = df

    @property
    def all_preds(self):
        return pd.concat(list(self.preds.values())) if self.preds else pd.DataFrame()