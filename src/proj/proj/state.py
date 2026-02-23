import pandas as pd
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
        assert isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be get from {owner}.{self.name}'
        return value

class _Account:
    """custom class to record account instance"""
    def __init__(self):
        self.name = 'account'
        _project_states[self.name] = None

    def __set__(self , instance, value):
        from src.res.factor.util import PortfolioAccount
        assert value is None or isinstance(value , PortfolioAccount) , f'value is not a PortfolioAccount instance: {type(value)} , cannot be set to {instance.__name__}.{self.name}'
        _project_states[self.name] = value

    def __get__(self , instance, owner):
        value = _project_states.get(self.name , None)
        assert isinstance(value , pd.DataFrame) , f'value is not a {pd.DataFrame} instance: {type(value)} , cannot be get from {owner}.{self.name}'
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
        assert isinstance(value , self._assertion_class) , f'value is not a {self._assertion_class} instance: {type(value)} , cannot be get from {owner}.{self.name}'
        return value

class _ProjStatesMeta(type):
    """meta class of ProjStates, allow to set attributes of _meta_slots"""
    trainer = _Trainer()
    account = _Account()
    factor = _Factor()

    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class ProjStates(metaclass=_ProjStatesMeta):
    """
    custom class to record anything
    example:
        Proj.States.trainer = trainer # for src.res.model.util.classes.BaseTrainer
        Proj.States.account = account # for src.res.factor.util.agency.portfolio_accountant.PortfolioAccount
        Proj.States.factor = factor   # for src.res.factor.util.classes.StockFactor
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