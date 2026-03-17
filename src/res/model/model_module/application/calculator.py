from typing import Any

from src.proj import Proj , CALENDAR
from src.res.model.util import ModelPath , BatchData
from src.res.model.data_module import DataModule
from src.res.model.model_module.module import get_predictor_module

class ModelCalculator:
    '''for a model_name/model_path to get batch_data easily'''
    def __init__(self , model : str | ModelPath | Any):
        self.model_path = ModelPath(model)
        
        self.config = self.model_path.load_config()
        self.model = get_predictor_module(self.config)
        with Proj.Silence:
            self.data_module  = DataModule(self.config , 'both').load_data() 

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.model_path})'

    def calculate(self , date : int , model_num : int = 0 , model_submodel : str = 'best'):
        """calculate the model output (batch_data) for a given date"""
        assert CALENDAR.is_trade_date(date) , f'{date} is not a trading day'
        self._set_model_and_data(date , model_num , model_submodel)
        batch_input = self.data_module.get_batch_input_of_date(date).to(self.model.device)
        batch_output = self.model(batch_input)
        return BatchData(batch_input , batch_output)

    def __call__(self , date : int , model_num : int = 0 , model_submodel : str = 'best'):
        return self.calculate(date , model_num , model_submodel)

    @property
    def model_keys(self) -> tuple[int , int , str]:
        """model_num , model_date , model_submodel of the current model"""
        if not hasattr(self , '_model_keys'):
            self._model_keys = (-1 , -1 , 'best')
        return self._model_keys

    @model_keys.setter
    def model_keys(self , value : tuple[int , int , str]):
        self._model_keys = value

    def _get_model_date(self , date : int) -> int:
        """get the latest model date before the query date"""
        model_dates = self.model_path.model_dates[self.model_path.model_dates < date]
        assert len(model_dates) > 0 , f'no model date before {date}'
        return model_dates[-1]

    def _set_model_and_data(self , date : int , model_num : int = 0 , model_submodel : str = 'best'):
        """set model and data module based on query keys : date , model_num , model_submodel"""
        model_date = self._get_model_date(date)
        model_param = self.config.model_param[model_num]
        self.data_module.setup('predict' ,  model_param , model_date)
        if self.model_keys != (model_num , model_date , model_submodel):
            self.model.load_model(model_num , model_date , model_submodel , model_param = model_param)
            self.model_keys = (model_num , model_date , model_submodel)
        return self.model