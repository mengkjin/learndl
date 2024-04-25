import torch
from ..util.classes import BaseCallBack

class CudaEmptyCache(BaseCallBack):
    def __init__(self , batch_interval = 20) -> None:
        super().__init__()
        self._interval = batch_interval
    def on_train_batch_end(self , Mmod):
        if (Mmod.batch_idx + 1) % self._interval == 0 : torch.cuda.empty_cache()
    def on_validation_batch_end(self , Mmod):
        if (Mmod.batch_idx + 1) % self._interval == 0 : torch.cuda.empty_cache()
    def on_test_batch_end(self , Mmod):
        if (Mmod.batch_idx + 1) % self._interval == 0 : torch.cuda.empty_cache()
        