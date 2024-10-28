import torch
import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm
from typing import Literal , Optional

from .module_selector import get_predictor_module
from ..module.nn import NNPredictor
from ...util import TrainConfig
from ...data_module import DataModule
from ....basic.util import ModelPath , HiddenPath

class ModelHiddenExtractor:
    '''for a model to predict recent/history data'''
    def __init__(self , model_name : str , 
                 model_nums : Optional[list | np.ndarray | int] = None , 
                 model_submodels : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = None ,
                 backward_days = 300 , forward_days = 160):
        self.backward_days = backward_days
        self.forward_days  = forward_days
        self.model_path = ModelPath(model_name)
        self.config = TrainConfig.load_model(self.model_path , override={'env.short_test':False , 'train.dataloader.sample_method':'sequential'})
        self.model  = get_predictor_module(self.config)
        assert isinstance(self.model , NNPredictor) , self.model
        self.data   = DataModule(self.config , 'both').load_data()

        if model_nums is None:
            self.model_nums = self.model_path.model_nums
        else:
            self.model_nums = [model_nums] if isinstance(model_nums , int) else model_nums

        if model_submodels is None:
            self.model_submodels = self.model_path.model_submodels
        else:
            self.model_submodels = [model_submodels] if isinstance(model_submodels , str) else model_submodels

        if not np.isin(self.model_path.model_dates , self.data.model_date_list).all():
            print('Caution! Not all model dates are in data.model_date_list, possibly due to short_test!')
            self.model_dates = self.data.model_date_list
        else:
            self.model_dates = self.model_path.model_dates
    
    def model_iter(self , model_dates : Optional[list | np.ndarray | int] = None , update = True):
        if model_dates is None: 
            model_dates = self.model_dates
        else:
            if isinstance(model_dates , int): model_dates = [model_dates]
            model_dates = np.intersect1d(self.model_dates , model_dates)
        if update:
            model_iter = [(model_date , model_num , submodel) for (model_date , model_num , submodel) 
                        in product(model_dates[:-1] , self.model_nums , self.model_submodels)
                        if not HiddenPath.target_hidden_path(self.model_path.name , model_num , model_date , submodel).exists()]
            model_iter += list(product(model_dates[-1:] , self.model_nums , self.model_submodels))
        else:
            model_iter = list(product(model_dates , self.model_nums , self.model_submodels))
        return model_iter

    def extract_hidden(self , model_dates : Optional[list | np.ndarray | int] = None ,
                       verbose = True , update = True , overwrite = False):
        model_iter = self.model_iter(model_dates , update)
        with torch.no_grad():
            for model_date , model_num , submodel in model_iter:
                hidden_path = HiddenPath(self.model_path.name , model_num , submodel)
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
        desc = f'Extract {self.model_path.name}/{model_num}/{model_date}/{submodel}'
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
        cls , model_name : str , model_nums : Optional[list | np.ndarray | int] = None , 
        model_submodels : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = ['best'] , 
        update = True , overwrite = False):

        extractor = cls(model_name , model_nums , model_submodels)
        extractor.extract_hidden(update = update , overwrite = overwrite)
        print('-' * 80)
        return extractor