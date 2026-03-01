import torch
import numpy as np
import pandas as pd

from typing import Any , ClassVar , Literal

from src.proj import MACHINE , Logger , Proj , CALENDAR
from src.res.model.util import ModelConfig , PredictionModel
from src.res.model.data_module import DataModule
from src.res.model.model_module.module import get_predictor_module

class ModelPredictor:
    '''for a model to predict recent/history data'''
    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    def __init__(self , reg_model : PredictionModel , use_data : Literal['fit' , 'predict' , 'both'] = 'both'):
        self.reg_model = reg_model
        self.use_data : Literal['fit' , 'predict' , 'both'] = use_data

        self.model_name = self.reg_model.model_name
        self.model_submodel = self.reg_model.submodel

        if self.reg_model.num == 'all': 
            self.model_nums = self.reg_model.model_nums
        elif isinstance(self.reg_model.num , (int , str)): 
            self.model_nums = [int(self.reg_model.num)]
        else:
            self.model_nums = list(self.reg_model.num)

        self.config = ModelConfig.load_model(self.model_name)
        self.model = get_predictor_module(self.config)
        self.df = pd.DataFrame()

        self._current_update_dates = []
        self._current_deploy_dates = []

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(reg_model={str(self.reg_model)})'

    def __call__(self , date : int):
        if self.df.empty or date not in self.df['date'].unique():
            self.predict_dates([date])
        return self.df.query('date == @date')
    
    def update_preds(self , update = True , overwrite = False , start_dt = None , end_dt = None):
        '''get update dates and predict these dates'''
        assert update != overwrite , 'update and overwrite must be different here'
        
        dates = CALENDAR.slice(CALENDAR.diffs(self.reg_model.pred_target_dates , self.reg_model.pred_dates if update else []) , start_dt , end_dt)
        with Proj.Silence:
            self.predict_dates(dates)
        self.save_preds()
        self.deploy()

    def predict_dates(self , dates : np.ndarray | list[int]):
        '''predict recent days'''
        if len(dates) == 0: 
            return self
        dates = np.array(dates)
        use_data0 = 'both' if min(dates) <= CALENDAR.today(-100) else 'predict'
        use_data1 = self.use_data 
        self.data_module  = DataModule(self.config , use_data0 if self.use_data == 'both' else use_data1).load_data() 
        pred_dates = dates[dates <= max(self.data_module.test_full_dates)]
        if pred_dates.size == 0: 
            return self
        assert any(self.reg_model.model_dates < pred_dates.min()) , f'no model date before {pred_dates}'
        df_task = pd.DataFrame({'pred_dates' : pred_dates , 
                                'model_date' : [max(self.reg_model.model_dates[self.reg_model.model_dates < d]) for d in pred_dates] , 
                                'calculated' : 0})
        torch.set_grad_enabled(False)
        df_list = []
        
        for model_date , df_sub in df_task.query('calculated == 0').groupby('model_date'):
            for model_num in self.model_nums:
                model_param = self.config.model_param[model_num]
                assert isinstance(model_date , int) , model_date
                self.data_module.setup('predict' ,  model_param , model_date)
                model = self.model.load_model(model_num , model_date , self.model_submodel , model_param = model_param)
                
                tdates = self.data_module.model_test_dates
                within = np.isin(tdates , df_sub.query('calculated == 0')['pred_dates'])
                loader = self.data_module.predict_dataloader()

                for tdate , do_calc , batch_input in zip(tdates , within , loader):
                    if not do_calc or len(batch_input) == 0: 
                        continue
                    # secid = data_module.datas.secid[batch_input.i[:,0].cpu().numpy()]
                    secid = self.data_module.batch_secid(batch_input)
                    df = model(batch_input).pred_df(secid , tdate , colnames = self.model_name , model_num = model_num)
                    df_list.append(df)
                    df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1

        if df_list:
            self.df = pd.concat(df_list , axis = 0).groupby(['date','secid'])[self.model_name].mean().reset_index()
        else:
            self.df = pd.DataFrame()
        return self

    def save_preds(self , df : pd.DataFrame | None = None , overwrite = False , secid_col = SECID_COLS , date_col = DATE_COLS):
        new_df = df if isinstance(df , pd.DataFrame) else self.df
        if new_df.empty: 
            return self
        self._current_update_dates = []
        for date , subdf in new_df.groupby(date_col):
            new_df = subdf.drop(columns='date').set_index(secid_col)
            self.reg_model.save_pred(new_df , date , overwrite , indent = 2 , vb_level = 3)
            self._current_update_dates.append(date)
        return self

    @property
    def deploy_required(self) -> bool:
        return MACHINE.hfm_factor_dir is not None

    def deploy(self , overwrite = False):
        '''deploy df by day to class.destination'''
        if MACHINE.hfm_factor_dir is None: 
            return self
        try:
            path_deploy = MACHINE.hfm_factor_dir.joinpath(self.reg_model.pred_name)
            path_deploy.parent.mkdir(parents=True,exist_ok=True)
            if overwrite:
                dates = self.reg_model.pred_dates
            else:
                deployed_dates = [int(path.name.removesuffix('.txt').split('_')[-1]) for path in path_deploy.glob('*.txt')]
                dates = np.setdiff1d(self.reg_model.pred_dates , deployed_dates)
            self._current_deploy_dates = []
            for date in dates:
                df = self.reg_model.load_pred(date , vb_level = 99)
                df.to_csv(path_deploy.joinpath(f'{self.reg_model.pred_name}_{date}.txt') , sep='\t', index=False, header=False)
                self._current_deploy_dates.append(date)
        except OSError as e:
            Logger.error(f'{self.reg_model.pred_name} deploy error: {e}')

        return self
    
    def df_corr(self , df = None , window = 30 , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''prediction correlation of ecent days'''
        if df is None: 
            df = self.df
        if df is None: 
            return NotImplemented
        dates : Any = df[date_col].unique()
        dates = np.sort(dates)[-window:]
        df = df.query(f'{date_col} in @dates')
        assert isinstance(df , pd.DataFrame) , f'{type(df)} is not a DataFrame'
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()

    @classmethod
    def get_model(cls , model_name : str , use_data : Literal['fit' , 'predict' , 'both'] = 'both'):
        model = PredictionModel.SelectModels(model_name)[0]
        return cls(model , use_data)
    
    @classmethod
    def update(cls , model_name : str | None = None , start_dt = None , end_dt = None , indent : int = 0 , vb_level : int = 1):
        '''Update prediction factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        Logger.note(f'Update : {cls.__name__} since last update!' , indent = indent , vb_level = vb_level)
        if start_dt is not None or end_dt is not None:
            Logger.stdout(f'Update from {start_dt} to {end_dt}' , indent = indent + 1 , vb_level = vb_level)
        models = PredictionModel.SelectModels(model_name)
        if model_name is None: 
            Logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , indent = indent + 1 , vb_level = vb_level)
        for model in models:
            md = cls(model)
            md.update_preds(update = True , overwrite = False , start_dt = start_dt , end_dt = end_dt)
            if md._current_update_dates:
                Logger.success(f'Update model prediction for {model} , len={len(md._current_update_dates)}' , indent = 1 , vb_level = vb_level)
            else:
                Logger.skipping(f'Model prediction for {model} is up to date' , indent = 1 , vb_level = vb_level)
            if md.deploy_required:
                if md._current_deploy_dates:
                    Logger.success(f'Deploy model prediction for {model} , len={len(md._current_deploy_dates)}' , indent = 1 , vb_level = vb_level)
                else:
                    Logger.skipping(f'Model prediction for {model} is up to date' , indent = 1 , vb_level = vb_level)
        return md

    @classmethod
    def recalculate(cls , model_name : str | None = None , start_dt = None , end_dt = None , indent : int = 0 , vb_level : int = 1):
        """Recalculate all model predictions"""
        Logger.note(f'Recalculate : {cls.__name__} since last recalculation!' , indent = indent , vb_level = vb_level)
        if start_dt is not None or end_dt is not None:
            Logger.stdout(f'Recalculate from {start_dt} to {end_dt}' , indent = indent + 1 , vb_level = vb_level)
        models = PredictionModel.SelectModels(model_name)
        if model_name is None: 
            Logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , indent = indent + 1 , vb_level = vb_level)
        for model in models:
            md = cls(model)
            md.update_preds(update = False , overwrite = True , start_dt = start_dt , end_dt = end_dt)
            if md._current_update_dates:
                Logger.stdout(f'Finish recalculating model prediction for {model} , len={len(md._current_update_dates)}' , indent = indent + 1 , vb_level = vb_level)
            else:
                Logger.stdout(f'No new recalculating model prediction for {model}' , indent = indent + 1 , vb_level = vb_level)
            if md.deploy_required:
                if md._current_deploy_dates:
                    Logger.stdout(f'Finish deploying model prediction for {model} , len={len(md._current_deploy_dates)}' , indent = indent + 1 , vb_level = vb_level)
                else:
                    Logger.stdout(f'No new deploying model prediction for {model}' , indent = indent + 1 , vb_level = vb_level)
        return md
