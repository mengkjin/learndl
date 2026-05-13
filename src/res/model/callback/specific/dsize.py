from src.data import BlockLoader
from src.res.model.util import BaseCallBack

class SpecificCB_DSize(BaseCallBack):
    '''in _dsize model fill [size] in batch_input.kwargs'''
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.size_block = None

    def init_buffer(self):
        if self.size_block is None: 
            self.size_block = BlockLoader('models', 'tushare_cne5_exp', ['size']).load()
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

    