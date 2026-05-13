import torch

from src.res.model.util import BaseCallBack

class CudaEmptyCache(BaseCallBack):
    '''CudaEmptyCache every few batch (pretty slow)'''
    CB_KEY_PARAMS = ['batch_interval']
    def __init__(self , trainer , batch_interval = 20 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.batch_interval = batch_interval
        # 2.5s for 86 epochs
    def empty_cache(self):
        if (self.trainer.batch_idx + 1) % self.batch_interval == 0 : 
            torch.cuda.empty_cache()
    def on_train_batch_end(self):        
        self.empty_cache()
    def on_validation_batch_end(self):   
        self.empty_cache()
    def on_test_batch_end(self):         
        self.empty_cache()
