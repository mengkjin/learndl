import torch
from typing import Any

from src.proj import Logger , Proj
from src.res.model.util import BasePredictorModel , BatchInput

class NullPredictor(BasePredictorModel):
    """null predictor for factor / db module (just pass through the data)"""
    AVAILABLE_CALLBACKS = ['BasicTestResult' , 'DetailedAlphaAnalysis' , 'StatusDisplay']

    def init_model(self , 
                   testor_mode : bool = False ,
                   *args , **kwargs):
        if testor_mode: 
            self._model_num , self._model_date , self._model_submodel = 0 , 0 , '0'
        self.model_dict.reset()
        self.complete_model_param = {}
        return self

    def new_model(self , *args , **kwargs):
        '''call when fitting/testing new model'''
        return self.init_model(*args , **kwargs)
    
    def load_model(self , *args , **kwargs):
        '''call when testing new model'''
        return self.init_model(*args , **kwargs)
    
    def forward(self , batch_input : BatchInput | torch.Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
        if len(batch_input) == 0: 
            return None
        x = batch_input.x if isinstance(batch_input , BatchInput) else batch_input
        assert isinstance(x , torch.Tensor) , f'{type(x)} is not a torch.Tensor'
        if x.ndim > 2:
            x = x.squeeze()
        if x.ndim == 1:
            x = x[...,None]
        assert x.ndim == 2 and x.shape[1] == 1 , f'{x.shape} cannot be passed through null predictor'
        return x

    def fit(self):
        """db model does not have fit stage"""
        raise NotImplementedError('null model does not have fit stage')

    def test(self):
        '''test the model inside'''
        Logger.note(f'model {self.model_str} test start' , vb_level = Proj.vb.max)

        for _ in self.trainer.iter_model_submodels():
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

        Logger.note(f'model {self.model_str} test done' , vb_level = Proj.vb.max)

    def collect(self , *args):
        return self.model_dict
