from __future__ import annotations
import torch

from src.res.model.util import BaseCallBack

MemoryOptimizationOption: dict[str, bool] = {
    'cuda' : False,
}
class MemoryOptimization(BaseCallBack):
    """Empty Cuda Cache Every Few Batches (Pretty Slow)"""
    CB_KEY_PARAMS = ['batch_interval']
    TurnOn : bool = False
    def __init__(self , trainer , batch_interval = 20 , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.batch_interval = batch_interval
        self.options = MemoryOptimizationOption
        # 2.5s for 86 epochs
    def __bool__(self):
        return any(self.options.values())
    @property
    def option_cuda(self):
        return self.options['cuda']
    def empty_cache(self):
        if self.option_cuda and (self.trainer.batch_idx + 1) % self.batch_interval == 0 : 
            torch.cuda.empty_cache()
    def on_train_batch_end(self):        
        self.empty_cache()
    def on_validation_batch_end(self):   
        self.empty_cache()
    def on_test_batch_end(self):         
        self.empty_cache()
