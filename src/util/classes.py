
from dataclasses import dataclass
from inspect import currentframe
from torch import Tensor
from typing import Any , Literal

@dataclass
class BatchOutput:
    outputs : Tensor | tuple | list
    @property
    def pred(self) -> Tensor:
        return self.outputs[0] if isinstance(self.outputs , (list , tuple)) else self.outputs
    @property
    def hidden(self) -> Tensor | None:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            return self.outputs[1]
        else:
            return None
    @classmethod
    def empty(cls): return cls(Tensor().requires_grad_())

@dataclass
class BatchData:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : Tensor | tuple[Tensor] | list[Tensor]
    y       : Tensor 
    w       : Tensor | None
    i       : Tensor 
    valid   : Tensor 
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: self.x = self.x[0]
    def to(self , device = None): return self.__class__(**{k:self.send_to(v , device) for k,v in self.__dict__.items()})
    def cpu(self):  return self.__class__(**{k:self.send_to(v , 'cpu') for k,v in self.__dict__.items()})
    def cuda(self): return self.__class__(**{k:self.send_to(v , 'cuda') for k,v in self.__dict__.items()})
    @property
    def is_empty(self): return len(self.y) == 0
    @classmethod
    def send_to(cls , obj , des : Any | Literal['cpu' , 'cuda']) -> Any:
        if obj is None: return None
        elif isinstance(obj , Tensor):
            if des == 'cpu': return obj.cpu()
            elif des == 'cuda': return obj.cuda()
            elif callable(des): return des(obj) 
            else: return obj.to(des)
        elif isinstance(obj , (list , tuple)):
            return type(obj)([cls.send_to(o , des) for o in obj])
        else: raise TypeError(obj)

class BaseCallBack:
    def __init__(self , model_module) -> None:
        self.model_module = model_module
        self._assert_validity()
    def _infomation(self):
        return f'Apply Callback of {self.__class__.__name__}'
    def __call__(self , hook_name): self.__getattribute__(hook_name)()
    def on_configure_model(self): pass
    def on_summarize_model(self): pass
    def on_data_end(self): pass 
    def on_data_start(self): pass 
    def on_after_backward(self): pass 
    def on_after_fit_epoch(self): pass 
    def on_before_backward(self): pass 
    def on_before_fit_epoch(self): pass 
    def on_before_save_model(self): pass 
    def on_fit_end(self): pass 
    def on_fit_model_end(self): pass 
    def on_fit_model_start(self): pass 
    def on_fit_start(self): pass
    def on_test_batch(self): pass 
    def on_test_batch_end(self): pass
    def on_test_batch_start(self): pass 
    def on_test_end(self): pass 
    def on_test_model_end(self): pass 
    def on_test_model_start(self): pass 
    def on_test_model_type_end(self): pass 
    def on_test_model_type_start(self): pass 
    def on_test_start(self): pass 
    def on_train_batch(self): pass 
    def on_train_batch_end(self): pass 
    def on_train_batch_start(self): pass 
    def on_train_epoch_end(self): pass 
    def on_train_epoch_start(self): pass 
    def on_validation_batch(self): pass 
    def on_validation_batch_end(self): pass 
    def on_validation_batch_start(self): pass
    def on_validation_epoch_end(self): pass
    def on_validation_epoch_start(self): pass
    @classmethod
    def _possible_hooks(cls):
        return [x for x in dir(cls) if cls._possible_hook(x)]
    @classmethod
    def _possible_hook(cls , name):
        return name.startswith('on_') and callable(getattr(cls , name))
        # return ['self'] == [v.name for v in signature(getattr(cls , name)).parameters.values()]
    @classmethod
    def _assert_validity(cls):
        if BaseCallBack in cls.__bases__:
            base_hooks = BaseCallBack._possible_hooks()
            self_hooks = cls._possible_hooks()
            invalid_hooks = [x for x in self_hooks if x not in base_hooks]
            if invalid_hooks:
                print(f'Invalid Hooks of {cls.__name__} :' , invalid_hooks)
                print('Use _ or __ to prefix these class-methods')
                raise TypeError(cls)
            
class WithCallBack(BaseCallBack):
    def __enter__(self): 
        env = getattr(currentframe() , 'f_back')
        while env.f_code.co_name == '__enter__': env = getattr(env , 'f_back')
        assert env.f_code.co_name.startswith('on_') , env.f_code.co_name
        self.hook_name = env.f_code.co_name
        pass
    def __exit__(self): 
        pass