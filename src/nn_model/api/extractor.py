import os , torch
import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm
from typing import Any , Literal , Optional

from ..classes import BatchOutput
from ..trainer import ModelEnsembler , NetDataModule
from ..util import Deposition , Device , TrainConfig
from ...basic import RegModel , PATH

class HiddenExtractor:
    '''for a model to predict recent/history data'''
    def __init__(self , model_name : str , 
                 model_nums : Optional[list | np.ndarray | int] = None , 
                 model_types : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = None):
        
        self.reg_model = RegModel(model_name)
        self.model_name = self.reg_model.name 
        
        if model_nums is None:
            self.model_nums = self.reg_model.model_nums
        else:
            self.model_nums = [model_nums] if isinstance(model_nums , int) else model_nums

        if model_types is None:
            self.model_types = self.reg_model.model_types
        else:
            self.model_types = [model_types] if isinstance(model_nums , str) else model_types

        self.contents : dict[str,pd.DataFrame] = {}

        self.config     = TrainConfig.load(f'{PATH.model}/{self.reg_model.name}' , override={'short_test':False})
        self.deposition = Deposition(self.config)
        self.device     = Device()

        self.data = NetDataModule(self.config , 'both').load_data()
        self.target_path = os.path.join(PATH.hidden , self.reg_model.name)

        if not np.isin(self.reg_model.model_dates , self.data.model_date_list).all():
            print('Caution! Not all model dates are in data.model_date_list, possibly due to short_test!')
            self.model_dates = self.data.model_date_list
        else:
            self.model_dates = self.reg_model.model_dates

    def hidden_key(self , model_num , model_type , model_date) : 
        return f'hidden.{model_num}.{model_type}.{model_date}.feather'

    def deploy(self):
        '''deploy df in contents to target path'''
        os.makedirs(self.target_path , exist_ok=True)
        for hidden_key , hidden_df in self.contents.items():
            hidden_df.to_feather(os.path.join(self.target_path , hidden_key))
        return self

    def update_model_iter(self):

        model_iter = [(d , n , t) for (d , n , t) 
                      in product(self.model_dates[:-1] , self.model_nums , self.model_types)
                      if not os.path.exists(os.path.join(self.target_path , self.hidden_key(n , t , d)))]
        model_iter += list(product(self.model_dates[-1:] , self.model_nums , self.model_types))
        return model_iter
    
    def given_model_iter(self , model_dates : Optional[list | np.ndarray | int] = None):
        if model_dates is None:
            print('Input model_dates to extract.')
            return []
        
        if isinstance(model_dates , int): model_dates = [model_dates]
        model_dates = np.intersect1d(self.model_dates , model_dates)
        model_iter = list(product(model_dates , self.model_nums , self.model_types))
        return model_iter

    def extract_hidden(self , what : Literal['given' , 'update'] , model_dates : Optional[list | np.ndarray | int] = None ,
                       verbose = True , deploy = False):
        if what == 'given':
            model_iter = self.given_model_iter(model_dates)
        elif what == 'update':
            model_iter = self.update_model_iter()
        else:
            raise KeyError(what)

        os.makedirs(self.target_path , exist_ok=True)
        with torch.no_grad():
            for model_date , model_num , model_type in model_iter:
                hidden_key = self.hidden_key(model_num , model_type , model_date)
                hidden_df  = self.model_hidden(model_num , model_type , model_date , verbose = verbose)
                if deploy:
                    hidden_df.to_feather(os.path.join(self.target_path , hidden_key))
                else:
                    self.contents[hidden_key] = hidden_df
        return self
    
    def model_hidden(self , model_num , model_type , model_date , verbose = True) -> pd.DataFrame:
        model_param = self.config.model_param[model_num]
        
        model = self.deposition.load_model(model_date , model_num , model_type)
        self.net = ModelEnsembler.get_net(self.config.model_module , model_param , model['state_dict'] , self.device)
        self.net.eval()

        df_list : list[pd.DataFrame] = []
        desc = f'Extract {model_num}/{model_type}/{model_date}' if verbose else ''

        self.data.setup('fit' ,  model_param , model_date)
        df_list += self.loader_hidden('train' , desc)
        df_list += self.loader_hidden('valid' , desc)

        self.data.setup('test' ,  model_param , model_date)
        df_list += self.loader_hidden('test' , desc)

        df = pd.concat(df_list , axis=0)
        return df
    
    def loader_hidden(self, dataset : Literal['train' , 'valid' , 'test'] , desc = ''):
        if dataset == 'train': loader = self.data.train_dataloader()
        elif dataset == 'valid': loader = self.data.val_dataloader()
        elif dataset == 'test': loader = self.data.test_dataloader()

        df_list : list[pd.DataFrame] = []
        if desc: 
            loader = tqdm(loader , total=len(loader))
            desc = f'{desc}/{dataset}'

        for batch_data in loader:
            batch_output = BatchOutput(self.net(batch_data.x))
            df = batch_output.hidden_df(batch_data , self.data.y_secid , self.data.y_date).assign(dataset = dataset)
            df_list.append(df)
            if isinstance(loader , tqdm): loader.set_description(desc)
        return df_list
    
    @classmethod
    def extract_model_hidden(
        cls , model_name : str , model_nums : Optional[list | np.ndarray | int] = None , 
        model_types : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = None):

        extractor = cls(model_name , model_nums , model_types)

        extractor.extract_hidden('update' , deploy = True)
        return extractor