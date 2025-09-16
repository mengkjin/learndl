import torch

from src.res.algo import AlgoModule
from src.res.model.util import BasePredictorModel , BatchData
from src.res.model.model_module.util.data_transform import batch_data_to_boost_input , batch_loader_concat , batch_data_flatten_x

class BoostPredictor(BasePredictorModel):
    '''a group of ensemble models , of same net structure'''
    AVAILABLE_CALLBACKS = ['StatusDisplay' , 'DetailedAlphaAnalysis' , 'GroupReturnAnalysis']

    def init_model(self , 
                   model_module : str | None = None , 
                   model_param : dict | None = None ,
                   testor_mode : bool = False ,
                   *args , **kwargs):
        if testor_mode: 
            self._model_num , self._model_date , self._model_submodel = 0 , 0 , '0'
        module = model_module if model_module else self.config.model_booster_type
        param  = model_param  if model_param  else self.model_param
        cuda = self.device.is_cuda     if self.config else None
        seed = self.config.random_seed if self.config else None

        self.booster = AlgoModule.get_booster(module , param , cuda , seed , given_name = self.model_full_name ,
                                        optuna = self.config.model_booster_optuna , n_trials = self.config.model_booster_optuna_n_trials)

        self.model_dict.reset()
        self.metrics.new_model(param)
        return self
    
    def new_model(self , *args , **kwargs):
        '''call when fitting new model'''
        return self.init_model(*args , **kwargs)

    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when testing new model'''
        model_file = self.load_model_file(model_num , model_date , submodel)
        assert self.model_submodel == 'best' , f'{self.model_submodel} does not defined in {self.__class__.__name__}'

        self.init_model(*args , **kwargs)
        self.booster.load_dict(model_file['booster_dict'])
        return self
    
    def forward(self , batch_data : BatchData | torch.Tensor , *args , **kwargs): 
        '''model object that can be called to forward'''
        if len(batch_data) == 0: 
            return None
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
        if self.trainer.verbosity > 10: 
            print('model fit start')

        self.new_model()

        self.booster.import_data(train = self.train_boost_input() , valid = self.valid_boost_input()).fit(silent = True)

        for _ in self.trainer.iter_train_dataloader():
            self.batch_forward()
            self.batch_metrics()

        for _ in self.trainer.iter_val_dataloader():
            self.batch_forward()
            self.batch_metrics()

        if self.trainer.verbosity > 10: 
            print('model fit done')

    def test(self):
        if self.trainer.verbosity > 10: 
            print('model test start')

        for _ in self.trainer.iter_model_submodels():
            self.load_model(submodel=self.model_submodel)
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

        if self.trainer.verbosity > 10: 
            print('model test done')

    def collect(self , submodel = 'best' , *args):
        self.model_dict.booster_dict = self.booster.to_dict()
        return self.model_dict
    
    # additional actions at hook