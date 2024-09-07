from torch import nn , Tensor , set_grad_enabled
from typing import Any , Optional

from .swa import choose_swa_method
from .optimizer import Optimizer
from ...util import BatchData , Metrics , ModelDict
from ...util.classes import BasePredictorModel
from ....algo.nn import GetNN , get_multiloss_params

class NNPredictor(BasePredictorModel):
    '''a group of ensemble models , of same net structure'''        
    @classmethod
    def get_net(cls , model_module : str , model_param = {} , device : Any = None , state_dict : Optional[dict[str,Tensor]] = {}):
        net = GetNN(model_module).nn_module(**model_param)
        assert isinstance(net , nn.Module) , net.__class__
        if state_dict: net.load_state_dict(state_dict)
        return device(net) if callable(device) else net.to(device)

    def new_model(self , model_module : str | Any = None , model_param = {} , device = None):
        if model_module is None:
            model_module = self.config.model_module 
            model_param = self.model_param
            device = self.device
        self.net = self.get_net(model_module , model_param, device = device)
        self.net.eval()
        for submodel in self.submodels.values(): submodel.reset()
        return self
    
    def get_multiloss_params(self):
        return get_multiloss_params(self.net)
    
    def create_submodels(self , submodels : list[str] | Any = None , use_score = True , **kwargs):
        if submodels is None: 
            self.submodels = {}
        else:
            self.submodels = {sub:choose_swa_method(sub)(self.ckpt , use_score , **kwargs) for sub in submodels}
        return self
    
    def prepare_training(self , trainer , use_score = True , **kwargs):
        super().prepare_training(trainer)
        self.create_submodels(self.config.model_submodels , use_score , **kwargs)
        return self

    def load_model(self , training : bool , model_type = 'best' , lr_multiplier = 1. , *args , **kwargs):
        '''call when fitting/testing new model'''
        self.new_model()
        for submodel in self.submodels.values(): submodel.reset()
        self.trainer.metrics.new_model(self.model_param)

        if training:
            if self.trainer.if_transfer:
                prev_model_file = self.trainer.deposition.load_model(self.trainer.prev_model_date , self.trainer.model_num , model_type)
                transferred = prev_model_file.exists() and prev_model_file['state_dict']
            else:
                transferred = False
            if transferred: self.net.load_state_dict(prev_model_file['state_dict'])
            self.optimizer : Optimizer = Optimizer(self.net , self.config , transferred , lr_multiplier , trainer = self.trainer)
            self.ckpt.new_model(self.model_param , self.trainer.model_date)
        else:
            model_file = self.trainer.deposition.load_model(self.trainer.model_date , self.trainer.model_num , model_type)
            self.net.load_state_dict(model_file['state_dict'])
    
    def forward(self , batch_data : BatchData | Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
        if len(batch_data) == 0: return None
        x = batch_data.x if isinstance(batch_data , BatchData) else batch_data
        return self.net(x , *args , **kwargs)

    def fit(self):
        for _ in self.trainer.iter_fit_epoches():
            self.trainer.on_train_epoch_start()
            for _ in self.trainer.iter_train_dataloader():
                self.batch_forward()
                self.batch_metrics()
                self.batch_backward()
            self.trainer.on_train_epoch_end()

            self.trainer.on_validation_epoch_start()
            for _ in self.trainer.iter_val_dataloader():
                self.batch_forward()
                self.batch_metrics()
            self.trainer.on_validation_epoch_end()

    def test(self):
        '''test the model inside'''
        for _ in self.trainer.iter_model_types():
            self.load_model(False , self.model_type)
            for _ in self.trainer.iter_test_dataloader():
                self.batch_forward()
                # before this is warmup stage , only forward
                self.batch_metrics()

    def batch_metrics(self) -> None:
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
        if self.status.dataset == 'test' and self.trainer.batch_idx < self.trainer.batch_warm_up: return
        '''if net has multiloss_params , get it and pass to calculate_from_tensor'''
        self.metrics.calculate(self.status.dataset , **self.metric_kwargs()).collect_batch()

    def batch_backward(self) -> None:
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
        assert self.status.dataset == 'train' , self.status.dataset
        self.trainer.on_before_backward()
        self.optimizer.backward(self.metrics.output)
        self.trainer.on_after_backward()

    def collect(self , submodel = 'best' , *args):
        net = self.submodels[submodel].collect(self.trainer , *args)
        model_dict = ModelDict(state_dict = net.state_dict())
        return model_dict
    
    # additional actions at hook
    def on_fit_model_start(self):
        self.load_model(True)

    def on_train_epoch_start(self):
        self.net.train()
        set_grad_enabled(True)

    def on_train_epoch_end(self):
        self.optimizer.scheduler_step(self.status.epoch)
    
    def on_validation_epoch_start(self):
        self.net.eval()
        set_grad_enabled(False)
    
    def on_validation_epoch_end(self):
        for submodel in self.submodels.values():  submodel.assess(self.net , self.status.epoch , self.metrics)

    def on_test_model_start(self):
        set_grad_enabled(False)

    def on_before_save_model(self):
        self.net = self.net.cpu()