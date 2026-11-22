"""
Callback specifically for DSize module
"""
from __future__ import annotations
from src.data import BlockLoader
from src.res.model.util import BaseCallBack

__all__ = ['SpecificCB_DSize']

class SpecificCB_DSize(BaseCallBack):
    """In [_dsize], Fill [size] in batch_input.kwargs"""
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.size_block = None

    def init_buffer(self):
        if self.size_block is None: 
            self.size_block = BlockLoader('models', 'tushare_cne5_exp', ['size'] , vb_level = 'never').load()
        self.data.buffer['size'] = self.size_block.align(self.data.y_secid , self.data.y_date).values.squeeze()
    def auxiliary_batch_kwargs(self, batch, *, training):
        i0, i1 = batch.i[:, 0].cpu(), batch.i[:, 1].cpu()
        size = self.data.buffer['size'][i0, i1].reshape(-1, 1).nan_to_num(0).to(batch.y.device)
        return {'size': size}

    def fill_batch_data(self):
        self.batch_input.kwargs = self.auxiliary_batch_kwargs(self.batch_input, training=True)

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

    