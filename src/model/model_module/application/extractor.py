import torch
import numpy as np
import pandas as pd

from itertools import product
from typing import Literal , Optional

from src.basic import ModelPath , HiddenPath , HiddenExtractingModel
from src.model.util import TrainConfig
from src.model.data_module import DataModule
from src.model.model_module.module import NNPredictor , get_predictor_module

class ModelHiddenExtractor:
    '''for a model to predict recent/history data'''
    def __init__(self , model : HiddenExtractingModel , backward_days = 300 , forward_days = 160):
        self.hidden_model = model
        self.backward_days = backward_days
        self.forward_days  = forward_days
        self.config = TrainConfig.load_model(self.model_path , override={'env.short_test':False , 'train.dataloader.sample_method':'sequential'})
        self.model  = get_predictor_module(self.config)
        assert isinstance(self.model , NNPredictor) , self.model
        self.data   = DataModule(self.config , 'both').load_data()
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_model})'

    @property
    def model_path(self):
        return self.hidden_model.model_path
    
    @property
    def model_nums(self):
        if self.hidden_model.nums is None:
            return self.model_path.model_nums
        elif isinstance(self.hidden_model.nums , int):
            return [self.hidden_model.nums]
        else:
            return self.hidden_model.nums
        
    @property
    def model_submodels(self):
        if self.hidden_model.submodels is None:
            return self.model_path.model_submodels
        elif isinstance(self.hidden_model.submodels , str):
            return [self.hidden_model.submodels]
        else:
            return self.hidden_model.submodels
    
    @property
    def model_alias(self):
        if self.hidden_model.alias is None:
            return self.hidden_model.name
        else:
            return self.hidden_model.alias
        
    @property
    def model_dates(self):
        if not np.isin(self.model_path.model_dates , self.data.model_date_list).all():
            print('Caution! Not all model dates are in data.model_date_list, possibly due to short_test!')
            return self.data.model_date_list
        else:
            return self.model_path.model_dates

    def model_iter(self , model_dates : Optional[list | np.ndarray | int] = None , update = True):
        if model_dates is None: 
            model_dates = self.model_dates
        else:
            if isinstance(model_dates , int): model_dates = [model_dates]
            model_dates = np.intersect1d(self.model_dates , model_dates)
        if update:
            model_iter = [(model_date , model_num , submodel) for (model_date , model_num , submodel) 
                        in product(model_dates[:-1] , self.model_nums , self.model_submodels)
                        if not HiddenPath.target_hidden_path(self.model_alias , model_num , model_date , submodel).exists()]
            model_iter += list(product(model_dates[-1:] , self.model_nums , self.model_submodels))
        else:
            model_iter = list(product(model_dates , self.model_nums , self.model_submodels))
        return model_iter

    def extract_hidden(self , model_dates : Optional[list | np.ndarray | int] = None ,
                       verbose = True , update = True , overwrite = False):
        model_iter = self.model_iter(model_dates , update)
        with torch.no_grad():
            for model_date , model_num , submodel in model_iter:
                hidden_path = HiddenPath(self.model_alias , model_num , submodel)
                self.model_hidden(hidden_path , model_date , verbose , overwrite)
        return self
    
    def model_hidden(self , hidden_path : HiddenPath , model_date :int , verbose = True , overwrite = False) -> pd.DataFrame | None:
        model_num , submodel = hidden_path.model_num , hidden_path.submodel
        self.model.load_model(model_num , model_date , submodel)

        old_hidden_df , exclude_dates = None , None
        if not overwrite:
            _ , old_hidden_df = hidden_path.get_hidden_df(model_date , exact = True)
            if len(old_hidden_df):
                exclude_dates = old_hidden_df['date'].unique()

        self.data.setup('extract' ,  self.config.model_param[model_num] , model_date , self.backward_days , self.forward_days)
        loader = self.data.extract_dataloader().filter_dates(exclude_dates=exclude_dates).enable_tqdm(disable = not verbose)      
        desc = f'Extract {self.model_alias}/{model_num}/{model_date}/{submodel}'
        hiddens : list[pd.DataFrame] = []
        for batch_data in loader:
            df = self.model(batch_data).hidden_df(batch_data , self.data.y_secid , self.data.y_date)
            if df is not None: hiddens.append(df)
            loader.display('/'.join([desc , str(df['date'][0])]))

        if len(hiddens) == 0: return

        hidden_df = pd.concat(hiddens , axis=0)
        if old_hidden_df is None: 
            hidden_df = pd.concat([old_hidden_df , hidden_df] , axis=0).drop_duplicates(['secid','date'])
        self.hidden_df = hidden_df
        hidden_path.save_hidden_df(hidden_df , model_date)
    
    @classmethod
    def update_hidden(
        cls , model_name : str | None = None , update = True , overwrite = False):
        if model_name is None:
            print(f'model_name is None, update all hidden models')
            models = HiddenExtractingModel.MODELS()
        else:
            models = [HiddenExtractingModel(model_name)]
        [print(f'  -->  update hidden feature for {model}') for model in models]
        for model in models:
            extractor = cls(model)
            extractor.extract_hidden(update = update , overwrite = overwrite)
            print('-' * 80)
        return extractor