from typing import Any
class DataHook:
    @classmethod
    def on_before_batch_transfer(cls , obj : Any , *args , **kwargs):
        return getattr(obj , 'on_before_batch_transfer' , cls.empty)(*args , **kwargs)
    @classmethod
    def transfer_batch_to_device(cls , obj : Any, *args , **kwargs):
        return getattr(obj , 'transfer_batch_to_device' , cls.empty)(*args , **kwargs)
    @classmethod
    def on_after_batch_transfer(cls , obj : Any, *args , **kwargs):
        return getattr(obj , 'on_after_batch_transfer' , cls.empty)(*args , **kwargs)
    @staticmethod
    def empty(batch_data , *args , **kwargs): 
        return batch_data
            
class ModelHook:
    def __init__(self , ptimer = None) -> None:
        self.tm = ptimer if ptimer is not None else self.EmptyTM
    def hook(self , func):
        def wrapper(*args , **kwargs):
            with self.tm(func.__name__):
                func(*args , **kwargs)
                self._hook_call(args[0] , func.__name__)
        return wrapper
    def _hook_call(self , obj , hook_name) -> None:
        [getattr(cb , hook_name)(obj) for cb in obj.callbacks if hasattr(cb , hook_name)]

    class EmptyTM:
        def __init__(self , *args): pass
        def __enter__(self): pass
        def __exit__(self , *args): pass

class BaseCallBack:
    def __init__(self) -> None:
        pass

class DynamicDataLink(BaseCallBack):
    def on_train_epoch_start(self , model_mod , *args):
        getattr(model_mod.net , 'dynamic_data_assign' , lambda *x:None)(model_mod)
    def on_validation_epoch_start(self , model_mod , *args):
        getattr(model_mod.net , 'dynamic_data_assign' , lambda *x:None)(model_mod)
    def on_test_epoch_start(self , model_mod , *args):
        getattr(model_mod.net , 'dynamic_data_assign' , lambda *x:None)(model_mod)
    def on_before_save_model(self , model_mod , *args):
        getattr(model_mod.net , 'dynamic_data_unlink' , lambda *x:None)()
    
