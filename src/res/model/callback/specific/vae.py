import torch
from src.res.model.util import BaseCallBack

class SpecificCB_VAE(BaseCallBack):
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