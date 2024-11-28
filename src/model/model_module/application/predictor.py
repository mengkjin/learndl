import torch
import numpy as np
import pandas as pd

from contextlib import nullcontext
from typing import Any , ClassVar , Optional

from src.basic import SILENT , CALENDAR , RegisteredModel , JS_FACTOR_DESTINATION
from src.model.util import TrainConfig
from src.model.data_module import DataModule
from src.model.model_module.module import get_predictor_module

class ModelPredictor:
    '''for a model to predict recent/history data'''
    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    def __init__(self , reg_model : RegisteredModel):
        self.reg_model = reg_model

        self.model_name = self.reg_model.name
        self.model_submodel = self.reg_model.submodel

        if self.reg_model.num == 'all': 
            self.model_nums = self.reg_model.model_nums
        elif isinstance(self.reg_model.num , (int , str)): 
            self.model_nums = [int(self.reg_model.num)]
        else:
            self.model_nums = list(self.reg_model.num)

        self.config = TrainConfig.load_model(self.model_name , override={'env.verbosity':0})
        self.model = get_predictor_module(self.config)
        self.df = pd.DataFrame()

        self.dir_deploy = JS_FACTOR_DESTINATION


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(reg_model={str(self.reg_model)})'
    
    def update_preds(self , update = True , overwrite = False , silent = True):
        '''get update dates and predict these dates'''
        assert update != overwrite , 'update and overwrite must be different here'
        with SILENT if silent else nullcontext():   
            dates = CALENDAR.diffs(self.reg_model.pred_target_dates , self.reg_model.pred_dates if update else [])
            self.predict_dates(dates , deploy = True)

    def predict_dates(self , dates : np.ndarray | list[int] , deploy = True):
        '''predict recent days'''
        data_module  = DataModule(self.config , 'both' if min(dates) <= CALENDAR.today(-100) else 'predict').load_data() 
        pred_dates = dates[dates <= max(data_module.test_full_dates)]
        if pred_dates.size == 0: return self
        assert any(self.reg_model.model_dates < pred_dates.min()) , f'no model date before {pred_dates}'
        df_task = pd.DataFrame({'pred_dates' : pred_dates , 
                                'model_date' : [max(self.reg_model.model_dates[self.reg_model.model_dates < d]) for d in pred_dates] , 
                                'calculated' : 0})
        torch.set_grad_enabled(False)
        df_list = []
        
        for model_date , df_sub in df_task[df_task['calculated'] == 0].groupby('model_date'):
            for model_num in self.model_nums:
                model_param = self.config.model_param[model_num]
                # print(model_date , 'old' if (data is data_mod_old) else 'new') 
                assert isinstance(model_date , int) , model_date
                data_module.setup('predict' ,  model_param , model_date)
                model = self.model.load_model(model_num , model_date , self.model_submodel , model_param = model_param)
                
                tdates = data_module.model_test_dates
                within = np.isin(tdates , df_sub[df_sub['calculated'] == 0]['pred_dates'])
                loader = data_module.predict_dataloader()

                for tdate , do_calc , batch_data in zip(tdates , within , loader):
                    if not do_calc or len(batch_data) == 0: continue
                    secid = data_module.datas.secid[batch_data.i[:,0].cpu().numpy()]
                    df = model(batch_data).pred_df(secid , tdate , colnames = self.model_name , model_num = model_num)
                    df_list.append(df)
                    df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1

        if df_list:
            self.df = pd.concat(df_list , axis = 0).groupby(['date','secid'])[self.model_name].mean().reset_index()
        else:
            self.df = pd.DataFrame()
        if deploy: self.deploy()
        return self
    
    def deploy(self , df : Optional[pd.DataFrame] = None , overwrite = False , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''deploy df by day to class.destination'''
        if df is None: df = self.df
        if df.empty: return self

        for date , subdf in df.groupby(date_col):
            new_df = subdf.drop(columns='date').set_index(secid_col)
            self.reg_model.save_pred(new_df , date , overwrite)

            if self.dir_deploy is not None:
                path_deploy = self.dir_deploy.joinpath(self.reg_model.pred_name , f'{self.reg_model.pred_name}_{date}.txt')
                path_deploy.parent.mkdir(parents=True,exist_ok=True)
                if (not overwrite or not path_deploy.exists()):
                    new_df.to_csv(path_deploy, sep='\t', index=True, header=False)
        return self
    
    def df_corr(self , df = None , window = 30 , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''prediction correlation of ecent days'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        df = df[df[date_col] >= CALENDAR.today(-window)]
        assert isinstance(df , pd.DataFrame)
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()
    
    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , silent = True):
        '''Update pre-registered factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        models = RegisteredModel.SelectModels(model_name)
        if model_name is None: print(f'model_name is None, update all registered models (len={len(models)})')
        for model in models:
            md = cls(model)
            md.update_preds(update = update , overwrite = overwrite , silent = silent)
            print(f'  -->  Finish updating model prediction for {model}')
        return md
