from __future__ import annotations
import torch
from typing import Any

from src.res.model.util import PredictorModel , BatchInput

class NullPredictor(PredictorModel):
    """null predictor for factor / db module (just pass through the data)"""

    def init_model(self , 
                   testor_mode : bool = False ,
                   *args , **kwargs):
        if testor_mode: 
            self._model_num , self._model_date , self._model_submodel = 0 , 0 , '0'
        self.model_dict.reset()
        self.complete_model_param = {}
        return self

    def reload_model(self , *args , **kwargs):
        '''call when fitting/testing new model'''
        return self.init_model(*args , **kwargs)
    
    def load_model(self , *args , **kwargs):
        '''call when testing new model'''
        return self.init_model(*args , **kwargs)

    def ckpt_state_dict(self):
        '''state dict of model at epoch to be saved in checkpoint'''
        return {
            'epoch' : self.status.epoch,
            'phase' : self.status.phase,
        }

    def load_state_dict(self , state_dict : dict):
        return self
    
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

    def collect(self , *args):
        return self.model_dict
