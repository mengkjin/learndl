import torch
from typing import Any
from .base import CallBack
def get_nn_specific_cb(module : str) -> Any:
    if module == 'factor_vae':
        return SpecCB_factor_vae
    elif module.startswith('tra_') :
        return SpecCB_tra

class SpecCB_tra(CallBack):
    '''assign and unlink dynamic data in tra networks'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module , with_cb=False)
    def _net_method(self , key , *args , **kwargs): 
        if (method := getattr(self.module.net,key,None)): method(*args , **kwargs)
    def on_train_epoch_start(self):      self._net_method('dynamic_data_assign' , self.module)
    def on_validation_epoch_start(self): self._net_method('dynamic_data_assign' , self.module)
    def on_test_model_type_start(self):  self._net_method('dynamic_data_assign' , self.module)
    def on_before_save_model(self):      self._net_method('dynamic_data_unlink')

class SpecCB_factor_vae(CallBack):
    '''factor vae additional batch_data components (y and reparameterize noise)'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module , with_cb=False)
        self._manual_seed = 42 if self.config.random_seed is None else self.config.random_seed
        torch.manual_seed(self._manual_seed)
    def on_train_batch_start(self):
        factor_num = self.module.net.factor_num
        y = self.module.batch_data.y
        self.module.batch_data.kwargs = {
            'y': y , 'alpha_noise' : torch.randn_like(y) ,
            'factor_noise' : torch.randn((factor_num,)).to(self.module.batch_data.y) ,
        }
    