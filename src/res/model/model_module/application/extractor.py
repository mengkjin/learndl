import torch , time
import numpy as np
import pandas as pd

from itertools import product

from src.proj import SILENT
from src.basic import CALENDAR , HiddenPath , HiddenExtractingModel
from src.res.model.util import TrainConfig
from src.res.model.data_module import DataModule
from src.res.model.model_module.module import NNPredictor , get_predictor_module

class ModelHiddenExtractor:
    '''for a model to predict recent/history data'''
    def __init__(self , model : HiddenExtractingModel , backward_days = 300 , forward_days = 160):
        self.hidden_model = model
        self.backward_days = backward_days
        self.forward_days  = forward_days
        self.config = TrainConfig.load_model(self.model_path , override={'env.short_test':False , 'train.dataloader.sample_method':'sequential'})
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
            with SILENT:
                self.data = DataModule(self.config , 'both').load_data()
            self.data_loaded = True
            print(f'-->  Loaded model data for {self.hidden_name} successfully!')

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
                       update = True , overwrite = False , silent = False):
        model_iter = self.model_iter(model_dates , update)
        self._current_update_dates = []
        with torch.no_grad():
            for model_date , model_num , submodel in model_iter:
                hidden_path = HiddenPath(self.hidden_name , model_num , submodel)
                modified_time = hidden_path.last_modified_time(model_date)
                if CALENDAR.is_updated_today(modified_time):
                    time_str = time.strftime('%Y/%m/%d %H:%M:%S',time.strptime(str(modified_time) , '%Y%m%d%H%M%S'))
                    print(f'-->  Skipping: {hidden_path.hidden_key} already updated at {time_str}!')
                    continue
                self.model_hidden(hidden_path , model_date , overwrite , silent)
                self._current_update_dates.append(model_date)
        return self
    
    def model_hidden(self , hidden_path : HiddenPath , model_date :int , overwrite = False , silent = False) -> pd.DataFrame | None:
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
        loader = self.data.extract_dataloader().filter_dates(exclude_dates=exclude_dates).enable_tqdm(disable = silent)      
        # something wrong here , exclude_dates is not working
        desc = f'Extract {self.hidden_name}/{model_num}/{model_date}/{submodel}'
        hiddens : list[pd.DataFrame] = []
        for batch_data in loader:
            date  = self.data.batch_date(batch_data)
            secid = self.data.batch_secid(batch_data)
            df = self.model(batch_data).hidden_df(secid , date)
            if df is not None: 
                hiddens.append(df)
            loader.display(f'{desc}/{date[0]}')

        if len(hiddens) == 0: 
            return
        if old_hidden_df is not None:  
            hiddens.insert(0 , old_hidden_df)
        hidden_df = pd.concat(hiddens , axis=0).drop_duplicates(['secid','date']).sort_values(['secid','date'])
        self.hidden_df = hidden_df
        hidden_path.save_hidden_df(hidden_df , model_date)
    
    @classmethod
    def SelectModel(cls , model_name : str) -> 'ModelHiddenExtractor':
        return cls(HiddenExtractingModel(model_name))

    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , silent = False):
        
        models = HiddenExtractingModel.SelectModels(model_name)
        if model_name is None: 
            print(f'model_name is None, update all hidden models (len={len(models)})')
        for model in models:
            extractor = cls(model)
            extractor.extract_hidden(update = update , overwrite = overwrite , silent = silent)
            if extractor._current_update_dates:
                print(f'  -->  Finish updating hidden feature extraction for {model} , len={len(extractor._current_update_dates)}')
            else:
                print(f'  -->  No new updating hidden feature extraction for {model}')
        return extractor