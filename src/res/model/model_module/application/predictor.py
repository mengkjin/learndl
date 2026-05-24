from __future__ import annotations
import torch
import numpy as np
import pandas as pd

from functools import cached_property
from typing import Any , ClassVar

from src.proj import MACHINE , Logger , Proj , CALENDAR
from src.res.model.util import PredictorPath , ModelConfig , DataModule , BatchData
from src.res.model.model_module.module import get_predictor_module

class ArchivedPredictorModel:
    '''for a model to predict recent/history data'''
    SECID_COLS : ClassVar[str] = 'secid'
    DATE_COLS  : ClassVar[str] = 'date'

    def __init__(self , predictor_path : PredictorPath):
        self.path = predictor_path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(path={self.path})'

    def __call__(self , date : int):
        if self.cached_df.empty or date not in self.cached_df['date'].unique():
            self.predict_dates([date])
        return self.cached_df.query('date == @date')

    @property
    def model_name(self) -> str:
        return self.path.model_name

    @cached_property
    def cached_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    @cached_property
    def current_update_dates(self) -> list[Any]:
        return []

    @cached_property
    def current_deploy_dates(self) -> list[Any]:
        return []

    @cached_property
    def config(self) -> ModelConfig:
        return ModelConfig(self.model_name , stage=2 , resume=1).start_model()

    @cached_property
    def model(self):
        return get_predictor_module(self.config)

    @cached_property
    def model_dates(self) -> np.ndarray:
        return self.path.model_dates

    @cached_property
    def model_nums(self) -> np.ndarray:
        return self.path.model_nums

    @cached_property
    def model_submodels(self) -> np.ndarray:
        return self.path.model_submodels

    def load_data(self , min_date : int | None = None , max_date : int | None = None):
        updated = CALENDAR.updated()
        min_date = min_date or 20170101
        max_date = max_date or updated
        if min_date > CALENDAR.today(-100):
            use_data = 'predict'
        elif max_date < updated:
            use_data = 'fit'
        else:
            use_data = 'both'

        if not hasattr(self , 'data_module'):
            self.data_module = DataModule(self.config , use_data).load_data() 
        elif self.data_module.use_data != 'both' and self.data_module.use_data != use_data:
            self.data_module = DataModule(self.config , 'both').load_data() 
        return self
    
    def update_preds(self , update = True , overwrite = False , start = None , end = None):
        '''get update dates and predict these dates'''
        assert update != overwrite , 'update and overwrite must be different here'
        
        dates = CALENDAR.slice(CALENDAR.diffs(self.path.pred_target_dates , self.path.pred_dates if update else []) , start , end)
        with Proj.silence:
            self.predict_dates(dates)
        self.save_preds()
        self.deploy()

    def batch_data(self , date : int , model_num : int , submodel : str , model_date : int | None = None):
        """calculate the batch data of a given date"""
        assert model_num in self.model_nums , f'model_num {model_num} not in {self.model_nums}'
        assert submodel in self.model_submodels , f'submodel {submodel} not in {self.model_submodels}'
        if model_date is None:
            prev_model_dates = self.model_dates[self.model_dates < date]
            model_date = prev_model_dates[-1] if len(prev_model_dates) > 0 else self.model_dates[0]
        assert model_date is not None and model_date in self.model_dates , f'model_date {model_date} not in {self.model_dates}'
        model_param = self.config.model_param[model_num]
        with Logger.Timer('load data'):
            self.load_data(date)
            self.data_module.setup('retrospective' , model_param , date)
        with Logger.Timer('load model'):
            model = self.model.load_model(model_num , model_date , self.path.use_submodel , model_param = model_param , cache_model = True)
        with Logger.Timer('load dataloader'):
            self.dataloader = self.data_module.retrospective_dataloader()
        with Logger.Timer('load batch_input'):
            batch_input = self.dataloader.of_date(date)
        with Logger.Timer('predict'):
            batch_output = model(batch_input)
        return BatchData(batch_input , batch_output)

    def predict_dates(self , dates : np.ndarray | list[int]):
        '''predict recent days'''
        if len(dates) == 0: 
            return self
        dates = np.array(dates)
        self.load_data(dates.min())
        pred_dates = dates[dates <= max(self.data_module.test_full_dates)]
        if pred_dates.size == 0: 
            return self
        assert any(self.path.model_dates < pred_dates.min()) , f'no model date before {pred_dates}'
        df_task = pd.DataFrame({'pred_dates' : pred_dates , 
                                'model_date' : [max(self.path.model_dates[self.path.model_dates < d]) for d in pred_dates] , 
                                'calculated' : 0})
        torch.set_grad_enabled(False)
        df_list : list[pd.DataFrame] = []
        
        for model_date , df_sub in df_task.query('calculated == 0').groupby('model_date'):
            for model_num in self.path.use_model_nums:
                model_param = self.config.model_param[model_num]
                assert isinstance(model_date , int) , model_date
                self.data_module.setup('retrospective' ,  model_param , model_date)
                model = self.model.load_model(model_num , model_date , self.path.use_submodel , model_param = model_param)
                
                tdates = self.data_module.model_test_dates
                within = np.isin(tdates , df_sub.query('calculated == 0')['pred_dates'])
                loader = self.data_module.retrospective_dataloader()

                for tdate , do_calc , batch_input in zip(tdates , within , loader):
                    if not do_calc or len(batch_input) == 0: 
                        continue
                    df = model(batch_input).pred_df(batch_input.secid , tdate , colnames = self.model_name , model_num = model_num)
                    df_list.append(df)
                    df_task.loc[df_task['pred_dates'] == tdate , 'calculated'] = 1

        if df_list:
            self.cached_df = pd.concat(df_list , axis = 0).groupby(['date','secid'])[self.model_name].mean().reset_index()
        return self

    def save_preds(self , df : pd.DataFrame | None = None , overwrite = False , secid_col = SECID_COLS , date_col = DATE_COLS):
        if df is None:
            df = self.cached_df
        if df.empty: 
            return self
        for date , subdf in df.groupby(date_col):
            subdf = subdf.drop(columns='date').set_index(secid_col)
            self.path.save_pred(subdf , date , overwrite , indent = 2 , vb_level = 3)
            self.current_update_dates.append(date)
        return self

    @property
    def deploy_required(self) -> bool:
        return MACHINE.hfm_factor_dir is not None

    def deploy(self , overwrite = False):
        '''deploy df by day to class.destination'''
        if MACHINE.hfm_factor_dir is None: 
            return self
        try:
            path_deploy = MACHINE.hfm_factor_dir.joinpath(self.path.pred_name)
            path_deploy.parent.mkdir(parents=True,exist_ok=True)
            if overwrite:
                dates = self.path.pred_dates
            else:
                deployed_dates = [int(path.name.removesuffix('.txt').split('_')[-1]) for path in path_deploy.glob('*.txt')]
                dates = np.setdiff1d(self.path.pred_dates , deployed_dates)

            for date in dates:
                df = self.path.load_pred(date , vb_level = 'never')
                df.to_csv(path_deploy.joinpath(f'{self.path.pred_name}_{date}.txt') , sep='\t', index=False, header=False)
                self.current_deploy_dates.append(date)
        except OSError as e:
            Logger.error(f'{self.path.pred_name} deploy error: {e}')

        return self
    
    def df_corr(self , df = None , window = 30 , secid_col = SECID_COLS , date_col = DATE_COLS):
        '''prediction correlation of ecent days'''
        if df is None: 
            df = self.cached_df
        if df is None: 
            return NotImplemented
        dates : Any = df[date_col].unique()
        dates = np.sort(dates)[-window:]
        df = df.query(f'{date_col} in @dates')
        assert isinstance(df , pd.DataFrame) , f'{type(df)} is not a DataFrame'
        return df.pivot_table(values = self.model_name , index = secid_col , columns = date_col).fillna(0).corr()

    @classmethod
    def get_model(cls , model_name : str):
        model = PredictorPath.SelectModels(model_name)[0]
        return cls(model)

    @classmethod
    def update(cls , model_name : str | None = None , start = None , end = None , indent : int = 0 , vb_level : Any = 1):
        '''Update prediction factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        Logger.note(f'Update : {cls.__name__} since last update!' , indent = indent , vb_level = vb_level)
        if start is not None or end is not None:
            Logger.stdout(f'Update from {start} to {end}' , indent = indent + 1 , vb_level = vb_level)
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            Logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , indent = indent + 1 , vb_level = vb_level)
        for model in models:
            md = cls(model)
            md.update_preds(update = True , overwrite = False , start = start , end = end)
            if md.current_update_dates:
                Logger.success(f'Update model prediction for {model} , len={len(md.current_update_dates)}' , indent = 1 , vb_level = vb_level)
            else:
                Logger.skipping(f'Model prediction for {model} is up to date' , indent = 1 , vb_level = vb_level)
            if md.deploy_required:
                if md.current_deploy_dates:
                    Logger.success(f'Deploy model prediction for {model} , len={len(md.current_deploy_dates)}' , indent = 1 , vb_level = vb_level)
                else:
                    Logger.skipping(f'Model prediction for {model} is up to date' , indent = 1 , vb_level = vb_level)
        return md

    @classmethod
    def recalculate(cls , model_name : str | None = None , start = None , end = None , indent : int = 0 , vb_level : Any = 1):
        """Recalculate all model predictions"""
        Logger.note(f'Recalculate : {cls.__name__} since last recalculation!' , indent = indent , vb_level = vb_level)
        if start is not None or end is not None:
            Logger.stdout(f'Recalculate from {start} to {end}' , indent = indent + 1 , vb_level = vb_level)
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            Logger.stdout(f'model_name is None, update all prediction models (len={len(models)})' , indent = indent + 1 , vb_level = vb_level)
        for model in models:
            md = cls(model)
            md.update_preds(update = False , overwrite = True , start = start , end = end)
            if md.current_update_dates:
                Logger.stdout(f'Finish recalculating model prediction for {model} , len={len(md.current_update_dates)}' , indent = indent + 1 , vb_level = vb_level)
            else:
                Logger.stdout(f'No new recalculating model prediction for {model}' , indent = indent + 1 , vb_level = vb_level)
            if md.deploy_required:
                if md.current_deploy_dates:
                    Logger.stdout(f'Finish deploying model prediction for {model} , len={len(md.current_deploy_dates)}' , indent = indent + 1 , vb_level = vb_level)
                else:
                    Logger.stdout(f'No new deploying model prediction for {model}' , indent = indent + 1 , vb_level = vb_level)
        return md
