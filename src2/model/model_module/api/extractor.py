import torch
import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm
from typing import Literal , Optional

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
        
        if model_nums is None:
            self.model_nums = self.model_path.model_nums
        else:
            self.model_nums = [model_nums] if isinstance(model_nums , int) else model_nums

        if model_submodels is None:
            self.model_submodels = self.model_path.model_submodels
        else:
            self.model_submodels = [model_submodels] if isinstance(model_submodels , str) else model_submodels

        self.contents : dict[str,pd.DataFrame] = {}
        self.config = TrainConfig.load_model(self.model_path.name , override={'env.short_test':False})
        self.model  = NNPredictor().bound_with_config(self.config)
        self.data   = DataModule(self.config , 'both').load_data()

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

    def extract_hidden(self , what : Literal['given' , 'update'] , model_dates : Optional[list | np.ndarray | int] = None ,
                       verbose = True , deploy = False):
        if what == 'given':
            model_iter = self.given_model_iter(model_dates)
        elif what == 'update':
            model_iter = self.update_model_iter()
        else:
            raise KeyError(what)

        with torch.no_grad():
            for model_date , model_num , submodel in model_iter:
                hidden_path = HiddenPath(self.model_path.name , model_num , submodel)
                hidden_df  = self.model_hidden(model_num , model_date , submodel , verbose = verbose)
                if deploy:
                    hidden_path.save_hidden_df(hidden_df , model_date)
                else:
                    self.contents[hidden_path.hidden_key] = hidden_df
        return self
    
    def model_hidden(self , model_num : int , model_date :int , submodel : str, verbose = True) -> pd.DataFrame:
        model_param = self.config.model_param[model_num]
        self.model.load_model(model_num , model_date , submodel)

        hiddens : list[pd.DataFrame] = []
        desc = f'Extract {model_num}/{submodel}/{model_date}' if verbose else ''

        self.data.setup('fit' ,  model_param , model_date)
        hiddens += self.loader_hidden('train' , desc)
        hiddens += self.loader_hidden('valid' , desc)

        self.data.setup('test' ,  model_param , model_date)
        hiddens += self.loader_hidden('test' , desc)

        df = pd.concat(hiddens , axis=0)
        return df
    
    def loader_hidden(self, dataset : Literal['train' , 'valid' , 'test'] , desc = ''):
        if dataset == 'train': loader = self.data.train_dataloader()
        elif dataset == 'valid': loader = self.data.val_dataloader()
        elif dataset == 'test': loader = self.data.test_dataloader()

        hiddens : list[pd.DataFrame] = []
        if desc: 
            loader = tqdm(loader , total=len(loader))
            desc = f'{desc}/{dataset}'

        secid , date = self.data.y_secid , self.data.y_date
        for batch_data in loader:
            hiddens.append(self.model(batch_data).hidden_df(batch_data , secid , date , dataset = dataset))
            if isinstance(loader , tqdm): loader.set_description(desc)
        return hiddens
    
    @classmethod
    def extract_model_hidden(
        cls , model_name : str , model_nums : Optional[list | np.ndarray | int] = None , 
        submodels : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = None):

        extractor = cls(model_name , model_nums , submodels)

        extractor.extract_hidden('update' , deploy = True)
        return extractor