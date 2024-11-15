import torch
import numpy as np
import pandas as pd

from typing import Any , ClassVar , Optional

from ..module import get_predictor_module
from ...util import TrainConfig
from ...data_module import DataModule
from ....basic import PATH , SILENT , THIS_IS_SERVER , CALENDAR
from ....basic.util import RegisteredModel , REG_MODELS
from ....func import today , date_offset

class ModelPredictor:
    '''for a model to predict recent/history data'''
    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    def __init__(self , reg_model : RegisteredModel):
        self.reg_model = reg_model

        self.model_name = self.reg_model.name
        self.model_submodel = self.reg_model.submodel
        self.alias = self.reg_model.alias if self.reg_model.alias else self.model_name

        if self.reg_model.num == 'all': 
            self.model_nums = self.reg_model.model_nums
        elif isinstance(self.reg_model.num , (int , str)): 
            self.model_nums = [int(self.reg_model.num)]
        else:
            self.model_nums = list(self.reg_model.num)

        self.config = TrainConfig.load_model(self.model_name , override={'env.verbosity':0})
        self.model = get_predictor_module(self.config)
        self.df = pd.DataFrame()

        self.dir_save = PATH.preds.joinpath(self.alias)
        self.dir_save.mkdir(parents=True,exist_ok=True)

        self.dir_deploy = PATH.FACTOR_DESTINATION_SERVER if THIS_IS_SERVER else PATH.FACTOR_DESTINATION_LAPTOP
        if self.dir_deploy is not None: 
            self.dir_deploy = self.dir_deploy.joinpath(self.alias)
            self.dir_deploy.mkdir(parents=True,exist_ok=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(reg_model={str(self.reg_model)})'

    @classmethod
    def update_factors(cls , silent = True):
        '''Update pre-registered factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        for model in REG_MODELS:
            md = cls(model)
            with SILENT: md.update().deploy()
            print(f'Finish model [{model.name}] predicting!')
        print('-' * 80)
        return md

    def update(self , start_dt = -10 , end_dt = 20991231 , update = True):
        '''get update dates and predict these dates'''
        end_dt = min(end_dt , today())
        if update and self.reg_model.start_dt > 0:
            start_dt = self.reg_model.start_dt
        elif start_dt <= 0: 
            start_dt = today(start_dt)

        model_dates  = self.reg_model.model_dates 
        start_dt     = max(start_dt , int(date_offset(min(model_dates) ,1)))

        update_dates = CALENDAR.td_within(start_dt , end_dt)
        stored_dates = PATH.pred_dates(self.alias , start_dt , end_dt)

        self.predict(np.setdiff1d(update_dates , stored_dates))
        self.deploy()
        return self

    def predict(self , dates : np.ndarray | list[int]):
        '''predict recent days'''
        data_module  = DataModule(self.config , 'both' if min(dates) <= today(-100) else 'predict').load_data() 
        pred_dates = dates[dates <= max(data_module.test_full_dates)]
        
        df_task = pd.DataFrame({'pred_dates' : pred_dates , 
                                'model_date' : [max(self.reg_model.model_dates[self.reg_model.model_dates  < d]) for d in pred_dates] , 
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

        self.df = pd.concat(df_list , axis = 0).groupby(['date','secid'])[self.model_name].mean().reset_index()
        return self
    
    def deploy(self , df : Optional[pd.DataFrame] = None , overwrite = False , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''deploy df by day to class.destination'''
        if df is None: df = self.df
        if df.empty: return self

        for date , subdf in df.groupby(date_col):
            new_df = subdf.drop(columns='date').set_index(secid_col)
            PATH.pred_save(new_df , self.alias , date , overwrite)

            if self.dir_deploy is not None:
                path_deploy = self.dir_deploy.joinpath(f'{self.alias}_{date}.txt')
                if (not overwrite or not path_deploy.exists()):
                    new_df.to_csv(path_deploy, sep='\t', index=True, header=False)
        return self
    
    def df_corr(self , df = None , window = 30 , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''prediction correlation of ecent days'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        df = df[df[date_col] >= today(-window)]
        assert isinstance(df , pd.DataFrame)
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()
