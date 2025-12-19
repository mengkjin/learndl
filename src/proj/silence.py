class Silence_old:
    """Silence manager, used to silence most of the project's output , nested usage is not supported"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self): 
        self._silent : bool = False
        self._enable : bool = True
    def __repr__(self):
        return f'Silence(silent={self._silent} , enable={self._enable})'
    @property
    def silent(self): 
        return self._silent and self._enable
    def __bool__(self): 
        return self.silent
    def __enter__(self) -> None: 
        self._raw_silent = self._silent
        self._silent = True
    def __exit__(self , *args) -> None: 
        self._silent = self._raw_silent
    def disable(self): 
        self._enable = False
    def enable(self): 
        self._enable = True

SILENT = Silence_old()

class _SilenceSilent:
    """Silent silencer, used to silence most of the project's output , nested usage is supported"""
    def __get__(self , instance, owner):
        instance_list = getattr(owner, 'instance_list' , None)
        return instance_list and instance_list[-1].enable

class Silence:
    """Silence manager, can be used to silence most of the project's output , nested usage is supported"""
    instance_list = []
    silent = _SilenceSilent()
    def __init__(self , enable = True):
        self.enable = enable
    def __repr__(self):
        return f'{self.__class__.__name__}({self.silent})'
    def __bool__(self): 
        return self.silent
    def __enter__(self) -> None: 
        self.__class__.instance_list.append(self)
    def __exit__(self , *args) -> None: 
        self.__class__.instance_list.remove(self)