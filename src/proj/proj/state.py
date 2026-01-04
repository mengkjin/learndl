import pandas as pd
from pathlib import Path
from typing import Any

__all__ = ['ProjStates']

_project_states : dict[str, Any] = {

}

class _Trainer:
    """custom class to record trainer instance"""
    def __init__(self):
        self.name = 'trainer'
        _project_states[self.name] = None

    def __set__(self , instance, value):
        from src.res.model.util.classes import BaseTrainer
        self._assertion_class = BaseTrainer
        assert value is None or isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be set to {instance.__name__}.{self.name}'
        _project_states[self.name] = value

    def __get__(self , instance, owner):
        value = _project_states.get(self.name , None)
        assert value is None or isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be get from {owner}.{self.name}'
        return value

class _Account:
    """custom class to record account instance"""
    def __init__(self):
        self.name = 'account'
        _project_states[self.name] = None

    def __set__(self , instance, value):
        assert value is None or isinstance(value , pd.DataFrame) , f'value is not a {pd.DataFrame} instance: {type(value)} , cannot be set to {instance.__name__}.{self.name}'
        _project_states[self.name] = value

    def __get__(self , instance, owner):
        value = _project_states.get(self.name , None)
        assert value is None or isinstance(value , pd.DataFrame) , f'value is not a {pd.DataFrame} instance: {type(value)} , cannot be get from {owner}.{self.name}'
        return value
class _Factor:
    """custom class to record factor instance"""
    def __init__(self):
        self.name = 'factor'
        _project_states[self.name] = None

    def __set__(self , instance, value):
        from src.res.factor.util import StockFactor
        self._assertion_class = StockFactor
        assert value is None or isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be set to {instance.__name__}.{self.name}'
        _project_states[self.name] = value

    def __get__(self , instance, owner):
        value = _project_states.get(self.name , None)
        assert value is None or isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be get from {owner}.{self.name}'
        return value

class _Email_Attachments:
    """custom class to record email path"""
    def __init__(self):
        self.name = 'email_attachments'
        _project_states[self.name] = []
    def __get__(self , instance, owner) -> list[Path | str]:
        return _project_states[self.name]

class _Exit_Files:
    """custom class to record app exit files"""
    def __init__(self):
        self.name = 'exit_files'
        _project_states[self.name] = []
    def __get__(self , instance, owner) -> list[Path | str]:
        return _project_states[self.name]

class _Export_Html_Files:
    """custom class to record export html files"""
    def __init__(self):
        self.name = 'export_html_files'
        _project_states[self.name] = []
    def __get__(self , instance, owner) -> list[Path | str]:
        return _project_states[self.name]

class _ProjStatesMeta(type):
    """meta class of ProjStates, allow to set attributes of _meta_slots"""
    trainer = _Trainer()
    account = _Account()
    factor = _Factor()
    email_attachments = _Email_Attachments()
    exit_files = _Exit_Files()
    export_html_files = _Export_Html_Files()
    current_vb_level : int | None = None

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class ProjStates(metaclass=_ProjStatesMeta):
    """
    custom class to record anything
    example:
        Proj.States.trainer = trainer # for src.res.model.util.classes.BaseTrainer
        Proj.States.account = account # for pandas.DataFrame portfolio account
        Proj.States.factor = factor   # for src.res.factor.util.classes.StockFactor
        Proj.States.email_attachments.append(email_attachment) # for list[Path | str] email attachments
        Proj.States.exit_files.append(exit_file) # for list[Path | str] exit files
    """

    @classmethod
    def info(cls) -> list[str]:
        """return the machine info list"""
        names = ', '.join([key for key , value in _project_states.items() if value])
        return [
            f'Proj States    : {names}', 
        ]

    @classmethod
    def status(cls) -> dict:
        """return the machine status dict"""
        status = {}
        for name , obj in _project_states.items():
            if obj is not None or (isinstance(obj , list) and obj):
                status[name] = obj
        return status