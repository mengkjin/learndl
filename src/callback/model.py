from ..util.classes import BaseCallBack

class DynamicDataLink(BaseCallBack):
    def _assign(self , net , obj):
        if hasattr(net , 'dynamic_data_assign'): net.dynamic_data_assign(obj)
    def _unlink(self , net):
        if hasattr(net , 'dynamic_data_unlink'): net.dynamic_data_unlink()
    def on_train_epoch_start(self , Mmod):      self._assign(Mmod.net , Mmod)
    def on_validation_epoch_start(self , Mmod): self._assign(Mmod.net , Mmod)
    def on_test_epoch_start(self , Mmod):       self._assign(Mmod.net , Mmod)
    def on_before_save_model(self , Mmod):      self._unlink(Mmod.net)