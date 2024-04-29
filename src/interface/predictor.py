import os
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import ClassVar , Literal

from ..environ import DIR
from ..util import Device , TrainConfig
from ..classes import BatchOutput
from ..data.DataFetcher import DataFetcher
from ..module import DataModule
from ..model import model as MODEL
from ..func.date import today , date_offset

@dataclass
class Predictor:
    '''for a model to predict recent/history data'''
    model_name : str
    model_type : Literal['best' , 'swalast' , 'swabest'] = 'swalast'
    model_num  : int = 0
    alias : str | None = None
    df    : pd.DataFrame | None = None

    destination : ClassVar[str] = '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha'
    secid_col : ClassVar[str] = 'secid'
    date_col  : ClassVar[str] = 'date'

    def __post_init__(self):
        if self.alias is None: self.alias = self.model_name

    def deploy(self , df : pd.DataFrame | None = None , overwrite = False , secid_col = secid_col , date_col = date_col):
        '''deploy df by day to class.destination'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        os.makedirs(f'{self.destination}/{self.alias}' , exist_ok=True)
        for date , subdf in df.groupby(date_col):
            des_path = f'{self.destination}/{self.alias}/{self.alias}_{date}.txt'
            if overwrite or not os.path.exists(des_path):
                subdf.drop(columns='date').set_index(secid_col).to_csv(des_path, sep='\t', index=True, header=False)

    def get_df(self , start_dt = -10 , end_dt = 20991231):
        '''save recent prediction to self.df'''
        self.df = self.predict(start_dt= start_dt , end_dt = end_dt)
        return self

    def df_corr(self , df = None , window = 30 , secid_col = secid_col , date_col = date_col):
        '''prediction correlation of ecent days'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        df = df[df[date_col] >= today(-window)]
        assert isinstance(df , pd.DataFrame)
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()

    def write_df(self , path):
        '''write down prediction df'''
        assert isinstance(self.df , pd.DataFrame)
        self.df.to_feather(path)

    def predict(self , start_dt = -10 , end_dt = 20991231) -> pd.DataFrame:
        '''predict recent days'''
        if start_dt <= 0: start_dt = today(start_dt)

        model_path = f'{DIR.model}/{self.model_name}'
        device       = Device()
        model_config = TrainConfig.load(model_path)

        model_param = model_config.model_param[self.model_num]
        model_files = sorted([p for p in os.listdir(f'{model_path}/{self.model_num}') if p.endswith(f'{self.model_type}.pt')])
        model_dates = np.array([int(mf.split('.')[0]) for mf in model_files])

        start_dt = max(start_dt , int(date_offset(min(model_dates) ,1)))
        calendar = DataFetcher.load_target_file('information' , 'calendar')
        assert calendar is not None

        require_model_data_old = (start_dt <= today(-100))

        data_mod_old = DataModule(model_config , False).load_data() if require_model_data_old else None
        data_mod_new = DataModule(model_config , True).load_data() 

        end_dt = min(end_dt , max(data_mod_new.test_full_dates))
        pred_dates = calendar[(calendar['calendar'] >= start_dt) & (calendar['calendar'] <= end_dt) & (calendar['trade'])]['calendar'].values

        df_task = pd.DataFrame({
            'pred_dates' : pred_dates ,
            'model_date' : [max(model_dates[model_dates < d_pred]) for d_pred in pred_dates] ,
            'calculated' : 0 ,
        })

        torch.set_grad_enabled(False)
        df_list = []
        for data_mod in [data_mod_old , data_mod_new]:
            if data_mod is None: continue
            assert isinstance(data_mod , DataModule)
            for model_date , df_sub in df_task[df_task['calculated'] == 0].groupby('model_date'):
                print(model_date , 'old' if (data_mod is data_mod_old) else 'new') 
                assert isinstance(model_date , int) , model_date
                data_mod.setup('predict' ,  model_param , model_date)
                sd_path = f'{model_path}/{self.model_num}/{model_date}.{self.model_type}.pt'

                net = MODEL.new(model_config.model_module , model_param , torch.load(sd_path) , device)
                net.eval()

                loader = data_mod.predict_dataloader()
                secid  = data_mod.datas.secid
                tdates = data_mod.model_test_dates
                iter_tdates = np.intersect1d(df_sub['pred_dates'][df_sub['calculated'] == 0] , tdates)

                for tdate in iter_tdates:
                    batch_data = loader[np.where(tdates == tdate)[0][0]]

                    pred = BatchOutput(net(batch_data.x)).pred
                    if len(pred) == 0: continue
                    df = pd.DataFrame({'secid' : secid[batch_data.i[:,0].cpu().numpy()] , 'date' : tdate , 
                                        self.model_name : pred.cpu().flatten().numpy()})
                    df_list.append(df)
                    df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1
        torch.set_grad_enabled(True)
        del data_mod_new , data_mod_old
        return pd.concat(df_list , axis = 0)