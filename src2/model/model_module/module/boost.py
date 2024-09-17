from torch import Tensor
from typing import Any , Optional

from ..util.data_transform import batch_data_to_boost_input , batch_loader_concat , batch_data_flatten_x
from ...util import BatchData
from ...util.classes import BasePredictorModel
from ....algo import getter

class BoostPredictor(BasePredictorModel):
    '''a group of ensemble models , of same net structure'''
    AVAILABLE_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']

    def init_model(self , 
                   model_module : Optional[str] = None , 
                   model_param : Optional[dict] = None ,
                   *args , **kwargs):
        module = model_module if model_module else self.config.model_booster_type
        param  = model_param  if model_param  else self.model_param
        cuda = self.device.is_cuda()   if self.config else None
        seed = self.config.random_seed if self.config else None
        self.booster = getter.boost(module , param , cuda , seed)

        self.model_dict.reset()
        self.metrics.new_model(param)
        return self
    
    def new_model(self , *args , **kwargs):
        '''call when fitting new model'''
        return self.init_model()

    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when testing new model'''
        model_file = self.load_model_file(model_num , model_date , submodel)
        assert self.model_submodel == 'best' , f'{self.model_submodel} does not defined in {self.__class__.__name__}'

        self.init_model(*args , **kwargs)
        self.booster.load_dict(model_file['booster_dict'])
        return self
    
    def forward(self , batch_data : BatchData | Tensor , *args , **kwargs): 
        '''model object that can be called to forward'''
        if len(batch_data) == 0: return None
        x = batch_data_flatten_x(batch_data) if isinstance(batch_data , BatchData) else batch_data
        pred = self.booster(x , *args , **kwargs)
        return pred
    
    def train_boost_input(self):
        long_batch = batch_loader_concat(self.data.train_dataloader())
        return batch_data_to_boost_input(long_batch , self.data.y_secid , self.data.y_date)

    def valid_boost_input(self):
        long_batch = batch_loader_concat(self.data.val_dataloader())
        return batch_data_to_boost_input(long_batch , self.data.y_secid , self.data.y_date)
    
    def fit(self):
        self.new_model()
        self.booster.import_data(train = self.train_boost_input() , valid = self.valid_boost_input())
        self.booster.fit(silent = True)

        for _ in self.trainer.iter_train_dataloader():
            self.batch_forward()
            self.batch_metrics()

        for _ in self.trainer.iter_val_dataloader():
            self.batch_forward()
            self.batch_metrics()

    def test(self):
        for _ in self.trainer.iter_model_submodels():
            self.load_model(submodel=self.model_submodel)
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

    def collect(self , submodel = 'best' , *args):
        self.model_dict.booster_dict = self.booster.to_dict()
        return self.model_dict
    
    # additional actions at hook