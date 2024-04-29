from .base import BasicCallBack

class DynamicDataLink(BasicCallBack):
    '''assign/unlink dynamic data in tra networks'''
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        self._print_info()
    def _net_method(self , key , *args , **kwargs): 
        if (method := getattr(self.module.net,key,None)): method(*args , **kwargs)
    def on_train_epoch_start(self):      self._net_method('dynamic_data_assign' , self.module)
    def on_validation_epoch_start(self): self._net_method('dynamic_data_assign' , self.module)
    def on_test_model_type_start(self):  self._net_method('dynamic_data_assign' , self.module)
    def on_before_save_model(self):      self._net_method('dynamic_data_unlink')