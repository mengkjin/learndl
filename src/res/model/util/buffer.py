from typing import Any , Callable , Literal

class BaseBuffer:
    '''dynamic buffer space for some module to use (tra), can be updated at each batch / epoch '''
    def __init__(self , device : Callable | None = None , always_on_device = False) -> None:
        self.device = device
        self.always = always_on_device
        self.contents : dict[str,Any] = {}

        self.register_setup()
        self.register_update()

    def __getitem__(self , key): 
        return self.contents[key]
    def __setitem__(self , key , value): 
        self.contents[key] = value
    @staticmethod
    def none_wrapper(*args, **kwargs): 
        return {}

    def update(self , new = None):
        if new is not None: 
            if self.always and self.device is not None: 
                new = self.device(new)
            self.contents.update(new)
        return self
    
    def get(self , keys , default = None , keep_none = True):
        if hasattr(keys , '__len__'):
            result = {k:self.contents.get(k , default) for k in keys}
            if not keep_none: 
                result = {k:v for k,v in result.items() if v is not None}
        else:
            result = self.contents.get(keys , default)
        if not self.always and self.device is not None: 
            result = self.device(result)
        return result

    def process(self , stage : Literal['setup' , 'update'] , data_module):
        new = getattr(self , f'{stage}_wrapper')(data_module)
        if new is not None: 
            if self.always and self.device is not None: 
                new = self.device(new)
            self.contents.update(new)
        return self
    
    def reset(self): 
        self.contents = {}
    
    def register_setup(self) -> None: 
        ...
    def register_update(self) -> None: 
        ...