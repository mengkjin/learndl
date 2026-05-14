import torch
from src.res.model.util import BaseCallBack
    
class SpecificCB_TRA(BaseCallBack):
    '''In [TRA], Fill [y] [hist_loss] in batch_input.kwargs , Update [hist_loss] in data.buffer'''
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