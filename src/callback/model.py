from ..util.classes import BaseCallBack

class DynamicDataLink(BaseCallBack):
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        print(f'{self._infomation()}() , this is essential for TRA models!')
    def _assign(self):
        net = self.model_module.net
        if hasattr(net , 'dynamic_data_assign'): net.dynamic_data_assign(self.model_module)
    def _unlink(self):
        net = self.model_module.net
        if hasattr(net , 'dynamic_data_unlink'): net.dynamic_data_unlink()
    def on_train_epoch_start(self):      self._assign()
    def on_validation_epoch_start(self): self._assign()
    def on_test_model_type_start(self):  self._assign()
    def on_before_save_model(self):      self._unlink()