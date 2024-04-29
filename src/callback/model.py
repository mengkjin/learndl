from .base import BasicCallBack

class DynamicDataLink(BasicCallBack):
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        self._print_info()
    def _net_method(self , key , *args , **kwargs): 
        if (method := getattr(self.module.net,key,None)): method(*args , **kwargs)
    def _assign(self): self._net_method('dynamic_data_assign' , self.module)
    def _unlink(self): self._net_method('dynamic_data_unlink')
    def on_train_epoch_start(self):      self._assign()
    def on_validation_epoch_start(self): self._assign()
    def on_test_model_type_start(self):  self._assign()
    def on_before_save_model(self):      self._unlink()