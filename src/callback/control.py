import numpy as np
import pandas as pd

import time
import torch
from ..util.classes import BaseCallBack , WithCallBack

class CudaEmptyCache(BaseCallBack):
    def __init__(self , model_module , batch_interval = 20) -> None:
        super().__init__(model_module)
        self._interval = batch_interval
        # 2.5s for 86 epochs
        print(f'{self._infomation()}({batch_interval}) , this is very slow!')
    def _empty_cache(self):
        if (self.model_module.batch_idx + 1) % self._interval == 0 : torch.cuda.empty_cache()
    def on_train_batch_end(self):        self._empty_cache()
    def on_validation_batch_end(self):   self._empty_cache()
    def on_test_batch_end(self):         self._empty_cache()

class ProcessTimer(WithCallBack):
    def __init__(self , model_module) -> None:
        super().__init__(model_module)
        self.pt = {}
        print(f'{self._infomation()}()')
    def __enter__(self):
        super().__enter__()
        self.start_time = time.time()
    def __exit__(self): 
        if self.hook_name not in self.pt.keys(): self.pt[self.hook_name] = []
        self.pt[self.hook_name].append(time.time() - self.start_time)
    def on_summarize_model(self):
        tb = pd.DataFrame([[k , len(v) , np.sum(v) , np.mean(v)] for k,v in self.pt.items()] ,
                          columns = ['keys' , 'num_calls', 'total_time' , 'avg_time'])
        print(tb.sort_values(by=['total_time'],ascending=False))
    @classmethod
    def _possible_hooks(cls):
        return [x for x in dir(cls) if cls._possible_hook(x)]
    @classmethod
    def _possible_hook(cls , name):
        return name.startswith('on_') and callable(getattr(cls , name))
    @classmethod
    def _assert_validity(cls):
        if WithCallBack in cls.__bases__:
            base_hooks = WithCallBack._possible_hooks()
            self_hooks = cls._possible_hooks()
            invalid_hooks = [x for x in self_hooks if x not in base_hooks]
            if invalid_hooks:
                print(f'Invalid Hooks of {cls.__name__} :' , invalid_hooks)
                print('Use _ or __ to prefix these class-methods')
                raise TypeError(cls)