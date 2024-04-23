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

