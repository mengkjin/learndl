"""Nested context manager to suppress most stdout-style output when enabled."""

class _SilenceSilent:
    """Silent silencer, used to silence most of the project's output , nested usage is supported"""
    def __get__(self , instance, owner):
        instance_list = getattr(owner, 'instance_list' , None)
        return bool(instance_list) and instance_list[-1].enable

class Silence:
    """
    Silence manager, can be used to silence most of the project's output , nested usage is supported
    if the input is True (default), the output will be suppressed.
    """
    instance_list = []
    silent = _SilenceSilent()
    def __init__(self , enable = True):
        self.enable = enable
    def __call__(self , silent : bool | None = None , *args , **kwargs):
        if silent is not None:
            return self.__class__(silent)
        else:
            return self
    def __repr__(self):
        return f'{self.__class__.__name__}({self.silent})'
    def __bool__(self): 
        return self.silent
    def __enter__(self) -> None: 
        self.__class__.instance_list.append(self)
    def __exit__(self , *args) -> None: 
        self.__class__.instance_list.remove(self)