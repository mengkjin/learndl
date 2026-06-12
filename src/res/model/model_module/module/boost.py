"""
Boost predictor module for trainer
"""
from __future__ import annotations
import torch

from src.res.algo import AlgoModule
from src.res.model.util import PredictorModel , BatchInput
from src.res.model.model_module.util.data_transform import batch_data_to_boost_input , batch_loader_concat , batch_data_flatten_x

__all__ = ['BoostPredictor']

class BoostPredictor(PredictorModel):
    """a group of ensemble models , of same net structure"""
    def init_model(self , 
                   model_module : str | None = None , 
                   model_param : dict | None = None ,
                   testor_mode : bool = False ,
                   *args , **kwargs):
        if testor_mode: 
            self._model_num , self._model_date , self._model_submodel = 0 , 0 , '0'
        module = model_module if model_module else self.config.model_module
        param  = model_param  if model_param  else self.model_param
        cuda = self.device.is_cuda     if self.config else None
        seed = self.config.random_seed if self.config else None

        self.boost = AlgoModule.get_boost(
            module , param , cuda , seed , given_name = self.model_full_name ,
            optuna = self.config.boost_optuna , override_criterion = self.config.criterion_boost,
            n_trials = self.config.boost_optuna_trials)

        self.model_dict.reset()
        self.complete_model_param = param
        return self
    
    def reload_model(self , *args , **kwargs):
        """call when fitting new model"""
        return self.init_model(*args , **kwargs)

    def load_model(self , model_num = None , model_date = None , submodel = None , *args , cache_model = False , **kwargs):
        """call when testing new model"""
        model_file = self.load_model_file(model_num , model_date , submodel)
        assert self.model_submodel == 'best' , f'{self.model_submodel} does not defined in {self.__class__.__name__}'
        if not cache_model or self.current_model_file.model_path != model_file.model_path:
            self.init_model(*args , **kwargs)
            self.boost.load_dict(model_file['boost_dict'])
            self.current_model_file = model_file
        return self

    def ckpt_state_dict(self):
        """revert model to an earlier epoch drom checkpoint"""
        return {
            'epoch' : self.status.epoch,
            'phase' : self.status.phase,
        }

    def load_state_dict(self , state_dict : dict):
        return self
    
    def forward(self , batch_input : BatchInput | torch.Tensor , *args , **kwargs): 
        """model object that can be called to forward"""
        if len(batch_input) == 0: 
            return None
        x = batch_data_flatten_x(batch_input) if isinstance(batch_input , BatchInput) else batch_input
        pred = self.boost(x , *args , **kwargs)
        return pred
    
    def train_boost_input(self):
        long_batch = batch_loader_concat(self.data.train_dataloader())
        return batch_data_to_boost_input(long_batch , self.data.y_secid , self.data.y_date)

    def valid_boost_input(self):
        long_batch = batch_loader_concat(self.data.val_dataloader())
        return batch_data_to_boost_input(long_batch , self.data.y_secid , self.data.y_date)
    
    def fit(self):
        self.boost.import_data(train = self.train_boost_input() , valid = self.valid_boost_input()).fit(silent = True)

        for _ in self.trainer.iter_train_dataloader():
            self.batch_forward()
            self.batch_metrics()

        for _ in self.trainer.iter_val_dataloader():
            self.batch_forward()
            self.batch_metrics()

    def collect(self , submodel = 'best' , *args):
        self.model_dict.boost_dict = self.boost.to_dict()
        return self.model_dict
    
    # additional actions at hook