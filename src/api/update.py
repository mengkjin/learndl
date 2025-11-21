from src.proj import Logger , MACHINE
from src.basic import CALENDAR

from .data import DataAPI
from .factor import FactorAPI
from .model import ModelAPI
from .trading import TradingAPI
from .notification import NotificationAPI

class UpdateAPI:
    @classmethod
    def daily(cls):
        if not MACHINE.updateable:
            Logger.cache_message('error', f'{MACHINE.name} is not updateable, skip rollback update')
            return
        DataAPI.update()
        if not DataAPI.is_updated():
            Logger.cache_message('error', 'Data is not updated to the latest date, skip model update')
            return
        FactorAPI.Factor.update(timeout = 3)
        ModelAPI.update()
        FactorAPI.Pooling.update(timeout = 3)
        FactorAPI.Stats.update(timeout = 3)
        FactorAPI.Hierarchy.update()
        TradingAPI.update()
        NotificationAPI.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        if not MACHINE.updateable:
            Logger.cache_message('error', f'{MACHINE.name} is not updateable, skip rollback update')
            return
        CALENDAR.check_rollback_date(rollback_date)
        DataAPI.rollback(rollback_date)
        FactorAPI.Factor.rollback(rollback_date , timeout = 10)
        FactorAPI.Pooling.rollback(rollback_date , timeout = 10)
        FactorAPI.Stats.rollback(rollback_date)
        FactorAPI.Hierarchy.rollback(rollback_date)

    @classmethod
    def weekly(cls):
        if not MACHINE.updateable:
            Logger.cache_message('error', f'{MACHINE.name} is not updateable, skip rollback update')
            return
        ModelAPI.update_models()
        
    