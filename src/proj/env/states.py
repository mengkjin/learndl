from pathlib import Path

class _ProjectStatesValue:
    """custom class to record a single value"""
    def __init__(self):
        self.name = self.__class__.__name__.removeprefix('_').lower()

    def __set__(self , instance, value):
        self.set_assertion_class()
        if self._assertion_class is not None:
            assert isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be set to {instance.__name__}.{self.name}'
        setattr(self , f'_{self.name}' , value)

    def __get__(self , instance, owner):
        obj = getattr(self , f'_{self.name}' , None)
        return obj

    def set_assertion_class(self) -> None:
        self._assertion_class = None

class _ProjectStatesList:
    """custom class to record a list of values"""
    def __init__(self):
        self.name = self.__class__.__name__.removeprefix('_').lower()

    def __set__(self , instance, value):
        raise AttributeError(f'cannot set {instance.__name__}.{self.name} to {value} , {self.name} is a list property')

    def __get__(self , instance, owner) -> list:
        if not hasattr(self , f'_{self.name}'):
            setattr(self , f'_{self.name}' , [])
        return getattr(self , f'_{self.name}')

class _Trainer(_ProjectStatesValue):
    """custom class to record trainer instance"""
    def set_assertion_class(self) -> None:
        from src.res.model.util.classes import BaseTrainer
        self._assertion_class = BaseTrainer

    def __get__(self , instance, owner):
        obj = super().__get__(instance, owner)
        assert obj is None or isinstance(obj , self._assertion_class) , f'value is not a BaseTrainer instance: {obj} , cannot be get from {owner}.{self.name}'
        return obj

class _Account(_ProjectStatesValue):
    """custom class to record account instance"""
    def set_assertion_class(self) -> None:
        import pandas as pd
        self._assertion_class = pd.DataFrame

    def __get__(self , instance, owner):
        obj = super().__get__(instance, owner)
        assert obj is None or isinstance(obj , self._assertion_class) , f'value is not a pd.DataFrame instance: {obj} , cannot be get from {owner}.{self.name}'
        return obj

class _Factor(_ProjectStatesValue):
    """custom class to record factor instance"""
    def set_assertion_class(self) -> None:
        from src.res.factor.util import StockFactor
        self._assertion_class = StockFactor

    def __get__(self , instance, owner):
        obj = super().__get__(instance, owner)
        assert obj is None or isinstance(obj , self._assertion_class) , f'value is not a StockFactor instance: {obj} , cannot be get from {owner}.{self.name}'
        return obj

class _Email_Attachments(_ProjectStatesList):
    """custom class to record email path"""
    def __get__(self , instance, owner) -> list[Path | str]:
        return super().__get__(instance, owner)

class _Exit_Files(_ProjectStatesList):
    """custom class to record app attachments"""
    def __get__(self , instance, owner) -> list[Path | str]:
        return super().__get__(instance, owner)

class _ProjStatesMeta(type):
    """meta class of ProjStates, allow to set attributes of _meta_slots"""
    trainer = _Trainer()
    account = _Account()
    factor = _Factor()
    email_attachments = _Email_Attachments()
    exit_files = _Exit_Files()

    _meta_slots = ['trainer', 'account', 'factor', 'email_attachments', 'exit_files']

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class ProjStates(metaclass=_ProjStatesMeta):
    """
    custom class to record anything
    example:
        ProjStates.trainer = trainer # for src.res.model.util.classes.BaseTrainer
        ProjStates.account = account # for pandas.DataFrame portfolio account
        ProjStates.factor = factor   # for src.res.factor.util.classes.StockFactor
        ProjStates.email_attachments.append(email_attachment) # for list[Path | str] email attachments
        ProjStates.exit_files.append(exit_file) # for list[Path | str] exit files
    """
    @classmethod
    def __setattr__(cls, key, value):
        if key not in cls._meta_slots:
            raise Exception(f'cannot set {cls.__name__}.{key} , only {cls._meta_slots} are allowed')
        object.__setattr__(cls, key, value)

    @classmethod
    def info(cls) -> list[str]:
        """Print the machine info"""
        names = ', '.join(cls._meta_slots)
        return [
            f'Proj States  : {names}', 
        ]

    @classmethod
    def status(cls) -> dict:
        """Print the machine info"""
        status = {}
        for name in cls._meta_slots:
            obj = getattr(cls, name)
            if obj is not None or (isinstance(obj , list) and obj):
                status[name] = obj
        return status