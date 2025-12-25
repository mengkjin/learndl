import sys

_config_slots = ['verbosity']

class _Verbosity:
    """
    custom verbosity
    example:
        ProjVerbosity.verbosity = 1 # for int verbosity
    """
    def __init__(self):
        self._verbosity = 2

    def __get__(self , instance, owner):
        return self._verbosity
    def __set__(self, instance, value):
        assert isinstance(value , int) , f'verbosity must be an integer , got {type(value)} : {value}'
        value = max(min(value , 10) , 0)
        if value != self._verbosity:
            sys.stderr.write(f'\u001b[31m\u001b[1mProject Verbosity Changed from {self._verbosity} to {value}\u001b[0m\n')
        else:
            sys.stderr.write(f'\u001b[31m\u001b[1mProject Verbosity Unchanged at {value}\u001b[0m\n')
        self._verbosity = value

class _ProjStatesMeta(type):
    """meta class of ProjConfig"""
    verbosity = _Verbosity()

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class ProjConfig(metaclass=_ProjStatesMeta):
    """
    custom class to record config
    example:
        ProjConfig.verbosity = 1 # for int verbosity
    """
    
    @classmethod
    def info(cls) -> list[str]:
        """return the machine info list"""
        return [
            f'Proj Verbosity : {cls.verbosity}', 
        ]

    @classmethod
    def status(cls) -> dict:
        """return the machine status dict"""
        status = {}
        for name in _config_slots:
            obj = getattr(cls, name)
            if obj is not None or (isinstance(obj , list) and obj):
                status[name] = obj
        return status
