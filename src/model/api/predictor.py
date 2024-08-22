import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any , ClassVar , Optional

from ..trainer import ModelEnsembler , NetDataModule
from ..util import BatchOutput , Deposition , Device , TrainConfig
from ...basic import RegModel , PATH , CONF , THIS_IS_SERVER , REG_MODELS , FACTOR_DESTINATION
from ...data import GetData
from ...func import today , date_offset

@dataclass
class Predictor:
    '''for a model to predict recent/history data'''
    reg_model : RegModel

    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    def __post_init__(self):
        self.model_name = self.reg_model.name
        self.model_type = self.reg_model.type
        self.alias = self.reg_model.alias if self.reg_model.alias else self.model_name

        if self.reg_model.num == 'all': 
            self.model_nums = self.reg_model.model_nums
        elif isinstance(self.reg_model.num , int): 
            self.model_nums = [self.reg_model.num]
        else:
            self.model_nums = list(self.reg_model.num)

        self.df = pd.DataFrame()

    @classmethod
    def update_factors(cls , silent = True):
        '''Update pre-registered factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        if THIS_IS_SERVER: return
        for model in REG_MODELS:
            md = cls(model)
            CONF.SILENT = True
            md.get_df().deploy()
            CONF.SILENT = False
            print(f'Finish model [{model.name}] predicting!')
        print('-' * 80)


    def deploy(self , df : Optional[pd.DataFrame] = None , overwrite = False , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''deploy df by day to class.destination'''
        if df is None: df = self.df
        if df is None: return NotImplemented
        FACTOR_DESTINATION.joinpath(self.alias).mkdir(exist_ok=True)
        for date , subdf in df.groupby(date_col):
            des_path = FACTOR_DESTINATION.joinpath(self.alias , f'{self.alias}_{date}.txt')
            if overwrite or not des_path.exists():
                subdf.drop(columns='date').set_index(secid_col).to_csv(des_path, sep='\t', index=True, header=False)

    def get_df(self , start_dt = -10 , end_dt = 20991231):
        '''save recent prediction to self.df'''
        self.df = self.predict(start_dt= start_dt , end_dt = end_dt)
        return self

    def df_corr(self , df = None , window = 30 , secid_col = SECID_COLS , date_col = DATE_COLS):
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

        device       = Device()
        model_config = TrainConfig.load(PATH.model.joinpath(self.model_name))
        deposition   = Deposition(model_config)
        model_dates  = self.reg_model.model_dates 
        start_dt     = max(start_dt , int(date_offset(min(model_dates) ,1)))

        data_mod = NetDataModule(model_config , 'both' if start_dt <= today(-100) else 'predict').load_data() 

        end_dt = min(end_dt , max(data_mod.test_full_dates))
        pred_dates = GetData.trade_dates(start_dt , end_dt)
        
        df_task = pd.DataFrame({'pred_dates' : pred_dates , 
                                'model_date' : [max(model_dates[model_dates < d]) for d in pred_dates] , 
                                'calculated' : 0})

        torch.set_grad_enabled(False)
        df_list = []
        
        for model_date , df_sub in df_task[df_task['calculated'] == 0].groupby('model_date'):
            for model_num in self.model_nums:
                model_param = model_config.model_param[model_num]
                # print(model_date , 'old' if (data is data_mod_old) else 'new') 
                assert isinstance(model_date , int) , model_date
                data_mod.setup('predict' ,  model_param , model_date)
                model = deposition.load_model(model_date , model_num , self.model_type)

                net = ModelEnsembler.get_net(model_config.model_module , model_param , model['state_dict'] , device)
                net.eval()

                net2 = ModelEnsembler.get_model(model_config , model , model_num = model_num , device = device)

                loader = data_mod.predict_dataloader()
                secid  = data_mod.datas.secid
                tdates = data_mod.model_test_dates
                iter_tdates = np.intersect1d(df_sub['pred_dates'][df_sub['calculated'] == 0] , tdates)

                for tdate in iter_tdates:
                    batch_data = loader[np.where(tdates == tdate)[0][0]]

                    pred = BatchOutput(net(batch_data.x)).pred
                    pred2 = net2(batch_data.x)

                    print(pred , pred2)

                    if len(pred) == 0: continue
                    df = pd.DataFrame({'model_num':model_num , 'date' : tdate , 
                                        'secid' : secid[batch_data.i[:,0].cpu().numpy()] , 
                                        self.model_name : pred.cpu().flatten().numpy()})
                    df_list.append(df)
                    df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1
        torch.set_grad_enabled(True)
        del data_mod
        df = pd.concat(df_list , axis = 0).groupby(['date','secid'])[self.model_name].mean().reset_index()
        return df
