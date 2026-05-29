from __future__ import annotations

import torch
import numpy as np
import pandas as pd

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any , Literal

from src.proj import CALENDAR , Proj , BaseClass , PATH , DB , Const
from src.res.model.util import ModelConfig , ModelPath , DataModule
from src.res.model.model_module.module import NNPredictor , get_predictor_module

class HiddenPath:
    """hidden factor path for nn models , used for extracting hidden states"""
    def __init__(self , model_name : str , model_num : int , submodel : str) -> None:
        self.model_name , self.model_num , self.submodel = model_name , model_num , submodel
        assert self.model_name in [p.name for p in PATH.hidden.iterdir()] , \
            f'Hidden path does not contains hidden data of {self.model_name}'

    def __repr__(self): 
        return f'{self.__class__.__name__}(model_name={self.model_name},model_num={self.model_num},submodel={self.submodel})'

    @classmethod
    def from_key(cls , hidden_key : str) -> HiddenPath:
        """
        create hidden path from hidden key
        example:
            HiddenPath.from_key('gru_day_V0.1.best')
        """
        model_name , model_num , submodel = cls.parse_hidden_key(hidden_key)
        return cls(model_name , model_num , submodel)

    @staticmethod
    def create_hidden_key(model_name : str , model_num : int , submodel : str) -> str:
        """
        create hidden key
        example:
            HiddenPath.create_hidden_key('gru_day_V0' , 1 , 'best') # gru_day_V0.1.best
        """
        return f'{model_name}.{model_num}.{submodel}'
    
    @property
    def hidden_key(self) -> str:
        """current hidden path's hidden key"""
        return self.create_hidden_key(self.model_name , self.model_num , self.submodel)
    
    @staticmethod
    def parse_hidden_key(hidden_key : str) -> tuple[str, int, str]:
        """parse hidden key"""
        model_name , model_num , submodel = hidden_key.split('.')
        assert submodel in ['best' , 'swabest' , 'swalast'] , f'{hidden_key} has invalid submodel: {submodel}'
        return model_name , int(model_num) , submodel
    
    @staticmethod
    def target_hidden_path(model_name : str , model_num : int , model_date , submodel : str) -> Path:
        """target hidden path"""
        return PATH.hidden.joinpath(model_name , str(model_num) , f'{model_date}.{submodel}.feather')
    
    def target_path(self , model_date: int) -> Path:
        """target hidden path of a given model date"""
        return self.target_hidden_path(self.model_name , self.model_num , model_date , self.submodel)
    
    def last_modified_date(self , model_date : int | None = None) -> int:
        """last modified date of the hidden path in '%Y%m%d' format"""
        if model_date is None: 
            model_dates = self.model_dates()
            model_date = int(model_dates.max()) if len(model_dates) else -1
        return PATH.file_modified_date(self.target_path(model_date))
    
    def last_modified_time(self , model_date : int | None = None) -> int:
        """last modified time of the hidden path in '%Y%m%d%H%M%S' format"""
        if model_date is None: 
            model_dates = self.model_dates()
            model_date = int(model_dates.max()) if len(model_dates) else -1
        return PATH.file_modified_time(self.target_path(model_date))

    def model_dates(self) -> np.ndarray:
        """model dates of source model"""
        suffix = f'.{self.submodel}.feather'
        parent = self.target_path(0).parent
        dates = [int(p.name.removesuffix(suffix)) for p in parent.iterdir() if p.name.endswith(suffix)]
        return np.sort(dates)

    def save_hidden_df(self , hidden_df : pd.DataFrame , model_date : int) -> None:
        """save hidden dataframe"""
        hidden_path = self.target_path(model_date)
        DB.save_df(hidden_df , hidden_path , overwrite = True , prefix = f'Hidden States' , indent = 1 , vb_level = 1)

    def get_hidden_df(self , model_date : int , exact = False) -> tuple[int, pd.DataFrame]:
        """get hidden dataframe"""
        if not exact: 
            model_date = self.closest_hidden_model_date(model_date)
        hidden_df = DB.load_df(self.target_path(model_date))
        return model_date , hidden_df
    
    def closest_hidden_model_date(self , model_date) -> int:
        """closest hidden model date"""
        possible_model_dates = self.model_dates()
        return possible_model_dates[possible_model_dates <= model_date].max()


class HiddenExtractionModel(ModelPath):
    '''
    for a hidden extraction model to extract hidden states
    model dict stored in configs/proj/model_settings.yaml file under hidden_extraction section
    '''
    MODEL_DICT : dict[str,dict[str,Any]] = Const.Model.strategies['hidden_extraction']
    def __new__(cls , *args , **kwargs) -> HiddenExtractionModel | Any:
        return super().__new__(cls , *args , **kwargs)

    def __init__(self , hidden_name : str , name: str | Any = None ,    
                 submodels : list | np.ndarray | Literal['best' , 'swalast' , 'swabest'] | None = None ,
                 nums : list | np.ndarray | int | None = None , assertion = True):
        if assertion:
            assert hidden_name in self.MODEL_DICT , f'{hidden_name} is not a hidden extraction model'
        if hidden_name in self.MODEL_DICT:
            reg_dict  = self.MODEL_DICT[hidden_name]
            name      = reg_dict['name']
            submodels = reg_dict['submodels']
            nums      = reg_dict['nums']

        self.hidden_name = hidden_name
        super().__init__(name)
        self.submodels = submodels
        self.nums = nums
        self.model_path = ModelPath(self.full_name)

    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(hidden_name={self.hidden_name},full_name={self.full_name},submodels={self.submodels},nums={str(self.nums)})'

    @classmethod
    def SelectModels(cls , hidden_names : list[str] | str | None = None) -> list[HiddenExtractionModel]:   
        """select hidden models"""
        if hidden_names is None: 
            hidden_names = list(cls.MODEL_DICT.keys())
        if isinstance(hidden_names , str): 
            hidden_names = [hidden_names]
        return [cls(key) for key in hidden_names]

class ModelHiddenExtractor(BaseClass.BoundLogger):
    '''for a model to predict recent/history data'''
    def __init__(self , model : HiddenExtractionModel , backward_days = 300 , forward_days = 160 , * , indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.hidden_model = model
        self.backward_days = backward_days
        self.forward_days  = forward_days
        self.config = ModelConfig(self.model_path , override={'train.dataloader.sample_method':'sequential'} , short_test = False , stage=2 , resume=1).start_model()
        self.model  = get_predictor_module(self.config)
        assert isinstance(self.model , NNPredictor) , self.model
        # self.load_model_data() # must load data before loading model, to get input_dim parameter
    
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
    def hidden_name(self):
        return self.hidden_model.hidden_name
        
    @property
    def model_dates(self):
        return self.model_path.model_dates
        
    def load_model_data(self):
        if not getattr(self , 'data_loaded' , False):
            with Proj.silence:
                self.data = DataModule(self.config , 'both').load_data()
            self.data_loaded = True
            self.logger.stdout(f'Load Model Data for Hidden Model {self.hidden_name} successfully!' , idt = 1 , vb = 1)

    def model_iter(self , model_dates : list | np.ndarray | int | None = None , update = True):
        if model_dates is None: 
            model_dates = self.model_dates
        else:
            if isinstance(model_dates , int): 
                model_dates = [model_dates]
            model_dates = np.intersect1d(self.model_dates , model_dates)
        if update:
            model_iter = [(model_date , model_num , submodel) for (model_date , model_num , submodel) 
                        in product(model_dates[:-1] , self.model_nums , self.model_submodels)
                        if not HiddenPath.target_hidden_path(self.hidden_name , model_num , model_date , submodel).exists()]
            model_iter += list(product(model_dates[-1:] , self.model_nums , self.model_submodels))
        else:
            model_iter = list(product(model_dates , self.model_nums , self.model_submodels))
        return model_iter

    def extract_hidden(self , model_dates : list | np.ndarray | int | None = None ,
                       update = True , overwrite = False):
        model_iter = self.model_iter(model_dates , update)
        self._current_update_dates = []
        with torch.no_grad():
            for model_date , model_num , submodel in model_iter:
                hidden_path = HiddenPath(self.hidden_name , model_num , submodel)
                modified_time = hidden_path.last_modified_time(model_date)
                if CALENDAR.is_updated_today(modified_time):
                    time_str = datetime.strptime(str(modified_time) , '%Y%m%d%H%M%S').strftime("%Y-%m-%d %H:%M:%S")
                    self.logger.skipping(f'{hidden_path.hidden_key} already updated at {time_str}!' , idt = 1 , vb = 1)
                    continue
                self.model_hidden(hidden_path , model_date , overwrite)
                self._current_update_dates.append(model_date)
        return self
    
    def model_hidden(self , hidden_path : HiddenPath , model_date :int , overwrite = False) -> pd.DataFrame | None:
        self.load_model_data()
        model_num , submodel = hidden_path.model_num , hidden_path.submodel
        self.model.load_model(model_num , model_date , submodel)

        old_hidden_df , exclude_dates = None , None
        if not overwrite:
            _ , old_hidden_df = hidden_path.get_hidden_df(model_date , exact = True)
            if len(old_hidden_df):
                exclude_dates = old_hidden_df['date'].unique()

        self.data.setup('extract' ,  self.config.model_param[model_num] , model_date , 
                        extract_backward_days = self.backward_days , extract_forward_days = self.forward_days)
        loader = self.data.extract_dataloader().filter_dates(exclude_dates=exclude_dates).enable_tqdm()      
        # something wrong here , exclude_dates is not working
        desc = f'Extract {self.hidden_name}/{model_num}/{model_date}/{submodel}'
        hiddens : list[pd.DataFrame] = []
        for batch_input in loader:
            df = self.model(batch_input).hidden_df(batch_input.secid , batch_input.date)
            if df is not None: 
                hiddens.append(df)
            loader.display(f'{desc}/{batch_input.date[0]}')

        if len(hiddens) == 0: 
            return
        if old_hidden_df is not None:  
            hiddens.insert(0 , old_hidden_df)
        hidden_df = pd.concat(hiddens , axis=0).drop_duplicates(['secid','date']).sort_values(['secid','date'])
        self.hidden_df = hidden_df
        hidden_path.save_hidden_df(hidden_df , model_date)

    @classmethod
    def SelectModel(cls , model_name : str) -> ModelHiddenExtractor:
        return cls(HiddenExtractionModel(model_name))

    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , indent : int = 0 , vb_level : Any = 1):
        cls.SetClassVB(vb_level , indent)
        cls.logger.note('Update since last update!')
        models = HiddenExtractionModel.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, update all hidden models (len={len(models)})' , idt = 1)
        for model in models:
            extractor = cls(model , indent = indent + 1 , vb_level = vb_level + 1)
            extractor.extract_hidden(update = update , overwrite = overwrite)
            if extractor._current_update_dates:
                extractor.logger.success(f'Update hidden feature extraction for {model} , len={len(extractor._current_update_dates)}')
            else:
                extractor.logger.skipping(f'Hidden feature extraction for {model} is up to date')
        return extractor