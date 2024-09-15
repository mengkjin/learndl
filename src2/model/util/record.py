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
        pred_idx = f'{trainer.model_num}.{trainer.model_type}.{trainer.model_date}.{trainer.batch_idx}'
        if pred_idx in self.preds.keys(): return
        if trainer.batch_idx < trainer.batch_warm_up: return
        
        which_output = trainer.model_param.get('which_output' , 0)
        ij = trainer.batch_data.i.cpu()
        secid , date = trainer.data.y_secid[ij[:,0]] , trainer.data.y_date[ij[:,1]]
        
        pred = trainer.batch_output.pred_df(secid , date).dropna()
        pred = pred.loc[pred['date'].isin(self.dates),:]
        if len(pred) == 0: return

        pred['model_num'] = trainer.model_num
        pred['model_type'] = trainer.model_type
 
        if which_output is None:
            pred['values'] = pred.loc[:,[col for col in pred.columns if col.startswith('pred.')]].mean(axis=1)
        else:
            pred['values'] = pred[f'pred.{which_output}']
        df = pred.loc[:,['model_num' , 'model_type' , 'secid' , 'date' , 'values']]

        self.preds[pred_idx] = df

    @property
    def all_preds(self):
        return pd.concat(list(self.preds.values()))