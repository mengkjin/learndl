import torch
from torch import set_grad_enabled
from typing import Any

from src.proj import Logger , Proj
from src.res.algo import AlgoModule
from src.res.model.util import BasePredictorModel , BatchData , BatchOutput , Optimizer
from src.res.model.model_module.util.swa import choose_swa_method
from src.res.model.model_module.util.data_transform import batch_data_to_boost_input , batch_loader_concat

class NNBooster(BasePredictorModel):
    '''a group of ensemble models , of same net structure'''    
    AVAILABLE_CALLBACKS = ['BasicTestResult' , 'DetailedAlphaAnalysis' , 'StatusDisplay']

    @property
    def booster_param(self): 
        assert self.config.model_booster_head , f'{self.config.model_booster_head} is not a valid booster head'
        return self.config.booster_head_param
    
    def init_model(self , 
                   model_nn_module : str | None = None , 
                   model_nn_param : dict | None = None , 
                   model_boost_module : str | None = None, 
                   model_boost_param : dict | None = None , 
                   testor_mode : bool = False ,
                   *args , **kwargs):
        if testor_mode: 
            self._model_num , self._model_date , self._model_submodel = 0 , 0 , '0'
        nn_module    = model_nn_module    if model_nn_module    else self.config.model_module
        nn_param     = model_nn_param     if model_nn_param     else self.model_param 
        boost_module = model_boost_module if model_boost_module else self.config.model_booster_head
        boost_param  = model_boost_param  if model_boost_param  else self.model_param

        device = self.config.device      if self.config else None
        cuda   = self.device.is_cuda     if self.config else None
        seed   = self.config.random_seed if self.config else None

        self.net = AlgoModule.get_nn(nn_module , nn_param , device)
        self.reset_submodels(*args , **kwargs)
        self.booster = AlgoModule.get_booster(boost_module , boost_param, cuda , seed , given_name = self.model_full_name)

        self.model_dict.reset()
        self.metrics.new_model(nn_param | boost_param)
        return self
    
    def reset_submodels(self , *args , **kwargs):
        if hasattr(self , 'submodels'):
            [submodel.reset() for submodel in self.submodels.values()]
        else:
            self.submodels = {sub:choose_swa_method(sub)(self.checkpoint ,*args , **kwargs) for sub in self.config.model_submodels}
        return self
    
    def new_model(self, lr_multiplier = 1. , *args , **kwargs):
        '''call when fitting new model'''
        self.init_model(*args , **kwargs)
        transferred = False
        if self.trainer and self.trainer.if_transfer:
            prev_model_file = self.deposition.load_model(self.model_num , self.trainer.prev_model_date)
            if prev_model_file.exists() and prev_model_file['state_dict']:
                self.net.load_state_dict(prev_model_file['state_dict'])
                transferred = True
        self.optimizer : Optimizer = Optimizer(self.net , self.config , transferred , lr_multiplier , trainer = self.trainer)
        self.checkpoint.new_model(self.model_param , self.model_date)
        
        return self

    def load_model(self , model_num = None , model_date = None , submodel = None , *args , **kwargs):
        '''call when fitting/testing new model'''
        model_file = self.load_model_file(model_num , model_date , submodel)
        assert self.model_submodel == 'best' , f'{self.model_submodel} does not defined in {self.__class__.__name__}'
        self.init_model(*args , **kwargs)
        self.net.load_state_dict(model_file['state_dict'])
        self.booster.load_dict(model_file['booster_head'])
        return self

    def multiloss_params(self): return AlgoModule.multiloss_params(self.net)

    def forward(self , batch_data : BatchData | torch.Tensor , *args , **kwargs) -> Any: 
        return self.forward_full(batch_data , *args , **kwargs)

    def forward_net(self , batch_data : BatchData | torch.Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
        if len(batch_data) == 0: 
            return None
        x = batch_data.x if isinstance(batch_data , BatchData) else batch_data
        return self.net(x , *args , **kwargs)
    
    def forward_full(self , batch_data : BatchData | torch.Tensor , *args , **kwargs): 
        '''model object that can be called to forward'''
        if len(batch_data) == 0: 
            return None
        hidden = BatchOutput(self.forward_net(batch_data , *args , **kwargs)).other['hidden']
        pred = self.booster(hidden , *args , **kwargs)
        return pred

    def train_boost_input(self):
        long_batch = batch_loader_concat(self.data.train_dataloader() , nn_to_calculate_hidden=self.net)
        return batch_data_to_boost_input(long_batch , self.data.y_secid , self.data.y_date)

    def valid_boost_input(self):
        long_batch = batch_loader_concat(self.data.val_dataloader() , nn_to_calculate_hidden=self.net)
        return batch_data_to_boost_input(long_batch , self.data.y_secid , self.data.y_date)
    
    def fit(self):
        Logger.note(f'model {self.model_str} fit start' , vb_level = Proj.vb.max)

        self.new_model()

        # fit net
        for _ in self.trainer.iter_fit_epoches():
            self.net.train()
            set_grad_enabled(True)
            for _ in self.trainer.iter_train_dataloader():
                self.batch_forward_net()
                self.batch_metrics()
                self.batch_backward()

            self.net.eval()
            set_grad_enabled(False)
            for _ in self.trainer.iter_val_dataloader():
                self.batch_forward_net()
                self.batch_metrics()

            for submodel in self.submodels.values():  
                submodel.assess(self.net , self.status.epoch , self.metrics)
            self.optimizer.scheduler_step(self.status.epoch)
        self.collect_net()

        # fit booster
        self.status.on_fit_model_start()
        self.booster.import_data(train = self.train_boost_input() , valid = self.valid_boost_input()).fit(silent = True)

        for _ in self.trainer.iter_train_dataloader():
            self.batch_forward()
            self.batch_metrics()

        for _ in self.trainer.iter_val_dataloader():
            self.batch_forward()
            self.batch_metrics()

        Logger.note(f'model {self.model_str} fit done' , vb_level = Proj.vb.max)

    def test(self):
        '''test the model inside'''
        Logger.note(f'model {self.model_str} test start' , vb_level = Proj.vb.max)

        for _ in self.trainer.iter_model_submodels():
            self.load_model(submodel=self.model_submodel)
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

        Logger.note(f'model {self.model_str} test done' , vb_level = Proj.vb.max)

    def batch_forward_net(self) -> None: 
        self.batch_output = BatchOutput(self.forward_net(self.batch_data))

    def collect_net(self , submodel = 'best' , *args):
        self.net = self.submodels[submodel].collect(self.trainer , *args)
        self.model_dict.state_dict = self.net.state_dict()
        return self.model_dict

    def collect(self , submodel = 'best' , *args):
        self.model_dict.booster_head = self.booster.to_dict()
        return self.model_dict
    
    # additional actions at hook
    def on_test_model_start(self):
        set_grad_enabled(False)

    def on_before_save_model(self):
        self.net = self.net.cpu()
