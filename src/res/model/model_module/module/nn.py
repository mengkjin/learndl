import torch
from torch import set_grad_enabled
from typing import Any

from src.proj import Logger , Proj
from src.res.algo import AlgoModule
from src.res.model.util import BasePredictorModel , BatchInput , Optimizer
from src.res.model.model_module.util.swa import choose_swa_method

class NNPredictor(BasePredictorModel):
    def init_model(self , 
                   model_module : str | None = None , 
                   model_param : dict | None = None , 
                   testor_mode : bool = False ,
                   *args , **kwargs):
        if testor_mode: 
            self._model_num , self._model_date , self._model_submodel = 0 , 0 , '0'
        module = model_module if model_module else self.config.model_module
        param  = model_param  if model_param  else self.model_param 

        device = self.config.device if self.config else None

        self.net = AlgoModule.get_nn(module , param , device)
        self.reset_submodels(*args , **kwargs)

        self.model_dict.reset()
        self.complete_model_param = param
        return self
    
    def reset_submodels(self , *args , **kwargs):
        if hasattr(self , 'submodels'):
            [submodel.reset() for submodel in self.submodels.values()]
        else:
            self.submodels = {sub:choose_swa_method(sub)(self.checkpoint ,*args , **kwargs) for sub in self.config.submodels}
        return self

    def new_model(self , lr_multiplier = 1. , *args , **kwargs):
        '''call when fitting/testing new model'''
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
        '''call when testing new model'''
        model_file = self.load_model_file(model_num , model_date , submodel)

        self.init_model(*args , **kwargs)
        self.net.load_state_dict(model_file['state_dict'])
        return self
    
    def forward(self , batch_input : BatchInput | torch.Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
        if len(batch_input) == 0: 
            return None
        x = batch_input.x if isinstance(batch_input , BatchInput) else batch_input
        return self.net(x , *args , **kwargs)

    def fit(self):
        Logger.note(f'model {self.model_str} fit start' , vb_level = Proj.vb.max)

        for _ in self.trainer.iter_fit_epoches():
            for _ in self.trainer.iter_train_dataloader():
                self.batch_forward()
                self.batch_metrics()
                self.batch_backward()

            for _ in self.trainer.iter_val_dataloader():
                self.batch_forward()
                self.batch_metrics()

        Logger.note(f'model {self.model_str} fit done' , vb_level = Proj.vb.max)

    def test(self):
        '''test the model inside'''
        Logger.note(f'model {self.model_str} test start' , vb_level = Proj.vb.max)

        for _ in self.trainer.iter_model_submodels():
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                self.batch_metrics()

        Logger.note(f'model {self.model_str} test done' , vb_level = Proj.vb.max)

    def collect(self , submodel = 'best' , *args):
        net = self.submodels[submodel].collect(self.trainer , *args)
        self.model_dict.state_dict = net.state_dict()
        return self.model_dict
    
    # additional actions at hook
    def on_train_epoch_start(self):
        self.net.train()
        set_grad_enabled(True)

    def on_train_epoch_end(self):
        self.optimizer.scheduler_step(self.status.epoch)
    
    def on_validation_epoch_start(self):
        self.net.eval()
        set_grad_enabled(False)
    
    def on_validation_epoch_end(self):
        for submodel in self.submodels.values():  
            submodel.assess(self.net , self.status.epoch , self.metrics)

    def on_test_model_start(self):
        set_grad_enabled(False)

    def on_before_save_model(self):
        self.net = self.net.cpu()
