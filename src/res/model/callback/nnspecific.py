import torch
from typing import Type

from src.res.algo import AlgoModule
from src.data import BlockLoader
from src.res.model.util import BaseCallBack

def specific_cb(module_name : str) -> Type[BaseCallBack] | None:
    nn_category = AlgoModule.nn_category(module_name)
    if module_name == 'gru_dsize':
        return SpecCB_DSize
    elif nn_category == 'vae':
        return SpecCB_VAE
    elif nn_category == 'tra':
        return SpecCB_TRA
    else:
        return None
    
class SpecCB_TRA(BaseCallBack):
    '''in TRA fill [y] [hist_loss] in batch_input.kwargs , update hist_loss in data.buffer'''
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        
    @property
    def net(self) -> torch.nn.Module: return getattr(self.trainer.model , 'net')
    def fill_batch_data(self):
        self.i0 = self.batch_input.i[:,0].cpu()
        self.i1 = self.batch_input.i[:,1].cpu()
        y = self.batch_input.y
        hl = self.data.buffer['hist_loss']
        rw = getattr(self.net , 'hist_loss_seq_len')
        hist_loss = torch.stack([hl[self.i0 , self.i1 + j + 1 - rw] for j in range(rw)],dim=-2)
        self.batch_input.kwargs = {'y': y , 'hist_loss' : hist_loss.to(y.device)}
    def init_buffer(self):
        hist_loss_shape = list(self.data.y_std.shape)
        hist_loss_shape[2] = getattr(self.net , 'num_states')
        self.data.buffer['hist_preds'] = torch.randn(hist_loss_shape)
        self.data.buffer['hist_loss']  = (self.data.buffer['hist_preds'] - self.data.y_std.nan_to_num(0)).square()
    def update_buffer(self):
        v0 : torch.Tensor = self.data.buffer['hist_preds'][self.i0 , self.i1]
        vp : torch.Tensor = self.model.batch_output.other['preds']
        vp = vp.detach().to(v0)
        self.data.buffer['hist_preds'][self.i0 , self.i1] = vp
        self.data.buffer['hist_loss'][self.i0 , self.i1] = (vp - v0).square()
    def on_fit_model_start(self):           
        self.init_buffer()
    def on_test_submodel_start(self):     
        self.init_buffer()
    def on_train_batch_start(self):         
        self.fill_batch_data()
    def on_validation_batch_start(self):    
        self.fill_batch_data()
    def on_test_batch_start(self):          
        self.fill_batch_data()
    def on_train_batch_end(self):           
        self.update_buffer()
    def on_validation_batch_end(self):      
        self.update_buffer()
    def on_test_batch_end(self):            
        self.update_buffer()

class SpecCB_VAE(BaseCallBack):
    '''in VAE fill [y] [alpha_noise] [factor_noise] in batch_input.kwargs'''
    CB_KEY_PARAMS = ['manual_seed']
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        torch.manual_seed(self.manual_seed) 

    @property
    def manual_seed(self) -> int: return 42 if self.trainer.config.random_seed is None else self.trainer.config.random_seed

    @property
    def net(self) -> torch.nn.Module: return getattr(self.trainer.model , 'net')
    
    def reparameterize(self , object_tensor : torch.Tensor , numel : int | None = None):
        if numel is None:
            return torch.randn_like(object_tensor)
        else:
            return torch.randn((numel,)).to(object_tensor)
        
    def on_train_batch_start(self):
        y = self.batch_input.y
        self.batch_input.kwargs = {
            'y': y , 'alpha_noise' : self.reparameterize(y) , 'factor_noise' : self.reparameterize(y , getattr(self.net , 'factor_num')) ,
        }

class SpecCB_DSize(BaseCallBack):
    '''in _dsize model fill [size] in batch_input.kwargs'''
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.size_block = None

    def init_buffer(self):
        if self.size_block is None: 
            self.size_block = BlockLoader('models', 'tushare_cne5_exp', ['size']).load().as_tensor()
        self.data.buffer['size'] = self.size_block.align(self.data.y_secid , self.data.y_date).values.squeeze()
    def fill_batch_data(self):
        i0 = self.batch_input.i[:,0].cpu()
        i1 = self.batch_input.i[:,1].cpu()
        size = self.data.buffer['size'][i0 , i1].reshape(-1,1).nan_to_num(0).to(self.batch_input.y.device)
        self.batch_input.kwargs = {'size': size}

    def on_fit_model_start(self):           
        self.init_buffer()
    def on_test_submodel_start(self):     
        self.init_buffer()
    def on_train_batch_start(self):         
        self.fill_batch_data()
    def on_validation_batch_start(self):    
        self.fill_batch_data()
    def on_test_batch_start(self):          
        self.fill_batch_data()

    