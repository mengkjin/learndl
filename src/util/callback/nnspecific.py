import torch
from typing import Any , Optional
from .base import CallBack
from src.nn import get_nn_category

def get_nn_specific_cb(module_name : str) -> Optional[Any]:
    nn_category = get_nn_category(module_name)
    if nn_category == 'vae':
        return SpecCB_VAE
    elif nn_category == 'tra':
        return SpecCB_TRA

class SpecCB_TRA2(CallBack):
    '''assign and unlink dynamic data in tra networks'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module , with_cb=False)
    def _net_method(self , key , *args , **kwargs): 
        if (method := getattr(self.module.net,key,None)): method(*args , **kwargs)
    def on_train_epoch_start(self):      self._net_method('dynamic_data_assign' , self.module)
    def on_validation_epoch_start(self): self._net_method('dynamic_data_assign' , self.module)
    def on_test_model_type_start(self):  self._net_method('dynamic_data_assign' , self.module)
    def on_before_save_model(self):      self._net_method('dynamic_data_unlink')

class SpecCB_TRA(CallBack):
    '''in TRA fill [y] [hist_loss] in batch_data.kwargs , update hist_loss in data.buffer'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module , with_cb=False)
    @property
    def i0(self): return self.module.batch_data.i[:,0]
    @property
    def i1(self): return self.module.batch_data.i[:,1]
    def fill_batch_data(self):
        hl = self.data.buffer['hist_loss']
        rolling_window = self.module.net.hist_loss_seq_len
        hist_loss = torch.stack([hl[self.i0 , self.i1 + j + 1 - rolling_window] 
                                 for j in range(rolling_window)],dim=-2)
        self.module.batch_data.kwargs = {'y': self.module.batch_data.y , 'hist_loss' : hist_loss}
    def init_buffer(self):
        hist_loss_shape = list(self.data.y.shape)
        hist_loss_shape[2] = self.module.net.num_states
        self.data.buffer['hist_preds'] = torch.randn(hist_loss_shape)
        self.data.buffer['hist_loss']  = (self.data.buffer['hist_preds'] - self.data.y.nan_to_num(0)).square()
    def update_buffer(self):
        v0 : torch.Tensor = self.data.buffer['hist_preds'][self.i0 , self.i1]
        vp : torch.Tensor = self.module.batch_output.other['preds']
        vp = vp.detach().to(v0)
        self.data.buffer['hist_preds'][self.i0 , self.i1] = vp
        self.data.buffer['hist_loss'][self.i0 , self.i1] = (vp - v0).square()
    def on_fit_model_start(self):           self.init_buffer()
    def on_test_model_type_start(self):     self.init_buffer()
    def on_train_batch_start(self):         self.fill_batch_data()
    def on_validation_batch_start(self):    self.fill_batch_data()
    def on_test_batch_start(self):          self.fill_batch_data()
    def on_train_batch_end(self):           self.update_buffer()
    def on_validation_batch_end(self):      self.update_buffer()
    def on_test_batch_end(self):            self.update_buffer()

class SpecCB_VAE(CallBack):
    '''in VAE fill [y] [alpha_noise] [factor_noise] in batch_data.kwargs'''
    def __init__(self , model_module) -> None:
        self._manual_seed = 42 if model_module.config.random_seed is None else model_module.config.random_seed
        super().__init__(model_module , with_cb=False)
        torch.manual_seed(self._manual_seed)
    def _reparameterize(self , object_tensor : torch.Tensor , numel : Optional[int] = None):
        if numel is None:
            return torch.randn_like(object_tensor)
        else:
            return torch.randn((numel,)).to(object_tensor)
    def on_train_batch_start(self):
        y = self.module.batch_data.y
        self.module.batch_data.kwargs = {
            'y': y , 'alpha_noise' : self._reparameterize(y) ,
            'factor_noise' : self._reparameterize(y , self.module.net.factor_num) ,
        }
    