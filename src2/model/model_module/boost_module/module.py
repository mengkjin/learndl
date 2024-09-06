from torch import Tensor
from typing import Any , Optional

from .data_transform import batch_data_to_boost_input , batch_loader_concat , batch_data_flatten_x
from ...util import BatchData , ModelDict
from ...util.classes import BasePredictorModel
from ....algo.boost import GeneralBooster , VALID_BOOSTERS


class BoostPredictor(BasePredictorModel):
    '''a group of ensemble models , of same net structure'''
    @classmethod
    def get_booster(cls , model_module : str , model_param = {} , cuda = False , seed = None , model_dict : Optional[dict] = None):
        assert model_module in VALID_BOOSTERS, model_module
        booster = GeneralBooster(model_module , model_param , cuda = cuda , seed = seed)
        if model_dict is not None: booster.load_dict(model_dict , cuda = cuda , seed = seed)
        return booster

    def new_model(self , model_module : str | Any = None , model_param = {} , device = None):
        if model_module is None:
            model_module = self.config.booster_type 
            model_param = self.model_param
            
        self.booster = self.get_booster(model_module , model_param, cuda=self.device.device.type == 'cuda' , seed = self.config.random_seed)
        return self

    def load_model(self , training : bool , model_type = 'best' , *args , **kwargs):
        '''call when fitting/testing new model'''
        self.new_model()
        self.trainer.metrics.new_model(self.model_param)
        assert model_type == 'best' , f'{model_type} does not defined in {self.__class__.__name__}'
        if training:
            ...
        else:
            model_file = self.trainer.deposition.load_model(self.trainer.model_date , self.trainer.model_num , model_type)
            self.booster.load_dict(model_file['booster_dict'])
        return self
    
    def forward(self , batch_data : BatchData | Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
        if len(batch_data) == 0: return None
        x = batch_data_flatten_x(batch_data) if isinstance(batch_data , BatchData) else batch_data
        return self.booster(x , *args , **kwargs)
    
    def fit(self):
        trainer = self.trainer
        train_batch = batch_loader_concat(self.data.train_dataloader())
        valid_batch = batch_loader_concat(self.data.val_dataloader())
        self.booster.import_data(train = batch_data_to_boost_input(train_batch , self.data.y_secid , self.data.y_date) , 
                                 valid = batch_data_to_boost_input(valid_batch , self.data.y_secid , self.data.y_date))
        
        trainer.on_train_epoch_start()
        for _ in trainer.iter_train_dataloader(given_loader=[train_batch]):
            self.booster.fit()
            self.batch_forward()
            self.batch_metrics()
        trainer.on_train_epoch_end()

        trainer.on_validation_epoch_start()
        for _ in trainer.iter_val_dataloader(given_loader=[valid_batch]):
            self.batch_forward()
            self.batch_metrics()
        trainer.on_validation_epoch_end()

    def test(self):
        trainer = self.trainer
        assert self.config.model_submodels == ['best'] , self.config.model_submodels
        for _ in trainer.iter_model_types():
            for _ in trainer.iter_test_dataloader():
                if trainer.batch_idx < trainer.batch_warm_up: continue
                self.batch_forward()
                self.batch_metrics()

    def batch_metrics(self) -> None:
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
        '''if net has multiloss_params , get it and pass to calculate_from_tensor'''
        self.metrics.calculate(self.status.dataset , **self.metric_kwargs()).collect_batch()
        
    def assess(self , *args , **kwargs): ...

    def collect(self , submodel = 'best' , *args):
        return ModelDict(booster_dict = self.booster.to_dict())
    
    # additional actions at hook
    def on_fit_model_start(self):
        self.load_model(True)

    def on_test_model_type_start(self):
        assert self.trainer.deposition.exists(self.trainer.model_date , self.trainer.model_num , self.trainer.model_type) , \
            (self.trainer.model_date , self.trainer.model_num , self.trainer.model_type)
        self.load_model(False , self.trainer.model_type)