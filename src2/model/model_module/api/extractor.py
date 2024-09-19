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
                 model_submodels : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = None):
        
        self.model_path = ModelPath(model_name)
        self.config = TrainConfig.load_model(self.model_path , override={'env.short_test':False , 'train.dataloader.sample_method':'sequential'})
        self.model  = get_predictor_module(self.config)
        assert isinstance(self.model , NNPredictor) , self.model
        self.data   = DataModule(self.config , 'both' , after_test_days = 40).load_data()

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

    def update_model_iter(self):
        model_iter = [(model_date , model_num , submodel) for (model_date , model_num , submodel) 
                      in product(self.model_dates[:-1] , self.model_nums , self.model_submodels)
                      if not HiddenPath.target_hidden_path(self.model_path.name , model_num , model_date , submodel).exists()]
        model_iter += list(product(self.model_dates[-1:] , self.model_nums , self.model_submodels))
        return model_iter
    
    def given_model_iter(self , model_dates : Optional[list | np.ndarray | int] = None):
        if model_dates is None:
            print('Input model_dates to extract.')
            return []
        
        if isinstance(model_dates , int): model_dates = [model_dates]
        model_dates = np.intersect1d(self.model_dates , model_dates)
        model_iter = list(product(model_dates , self.model_nums , self.model_submodels))
        return model_iter

    def extract_hidden(self , model_dates : Optional[list | np.ndarray | int] = None ,
                       verbose = True , update = True):
        model_iter = self.update_model_iter() if update else self.given_model_iter(model_dates)
        with torch.no_grad():
            for model_date , model_num , submodel in model_iter:
                hidden_path = HiddenPath(self.model_path.name , model_num , submodel)
                self.model_hidden(hidden_path , model_date , verbose , update)
        return self
    
    def model_hidden(self , hidden_path : HiddenPath , model_date :int , verbose = True , update = True) -> pd.DataFrame | None:
        model_num , submodel = hidden_path.model_num , hidden_path.submodel
        old_hidden_df = None 
        exclude_dates = None
        if update:
            _ , old_hidden_df = hidden_path.get_hidden_df(model_date , exact = True)
            if len(old_hidden_df):
                exclude_dates = old_hidden_df['date'].unique()
        
        model_param = self.config.model_param[model_num]
        self.model.load_model(model_num , model_date , submodel)

        hiddens : list[pd.DataFrame] = []
        desc = f'Extract {self.model_path.name}/{model_num}/{model_date}/{submodel}' if verbose else ''

        self.data.setup('fit' ,  model_param , model_date)
        hiddens += self.loader_hidden('train' , desc , exclude_dates)
        hiddens += self.loader_hidden('valid' , desc , exclude_dates)

        self.data.setup('test' ,  model_param , model_date)
        hiddens += self.loader_hidden('test' , desc , exclude_dates)

        if len(hiddens) == 0: return

        hidden_df = pd.concat(hiddens , axis=0)
        if old_hidden_df is None: 
            hidden_df = pd.concat([old_hidden_df , hidden_df] , axis=0).drop_duplicates(['secid','date','dataset'])
        self.hidden_df = hidden_df
        hidden_path.save_hidden_df(hidden_df , model_date)
    
    def loader_hidden(self, dataset : Literal['train' , 'valid' , 'test'] , desc = '' , exclude_dates = None):
        if dataset == 'train': loader = self.data.train_dataloader()
        elif dataset == 'valid': loader = self.data.val_dataloader()
        elif dataset == 'test': loader = self.data.test_dataloader()

        hiddens : list[pd.DataFrame] = []
        loader = loader.filter_dates(exclude_dates=exclude_dates)        
        if desc:  loader = loader.init_tqdm()

        for batch_data in loader:
            df = self.model(batch_data).hidden_df(batch_data , self.data.y_secid , self.data.y_date , dataset = dataset)
            if df is not None: hiddens.append(df)
            loader.display(f'{desc}/{dataset}/{df.date[0]}')
        return hiddens
    
    @classmethod
    def update_hidden(
        cls , model_name : str , model_nums : Optional[list | np.ndarray | int] = None , 
        model_submodels : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = ['best']):

        extractor = cls(model_name , model_nums , model_submodels)
        extractor.extract_hidden()
        return extractor