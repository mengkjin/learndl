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
        if not MACHINE.updatable:
            Logger.failure(f'{MACHINE.name} is not updatable, skip rollback update')
            return
        DataAPI.update()
        if not DataAPI.is_updated():
            Logger.failure('Data is not updated to the latest date, skip model update')
            return
        FactorAPI.Market.update()
        FactorAPI.Stock.update(timeout = 3)
        FactorAPI.Risk.update()
        ModelAPI.update()
        FactorAPI.Pooling.update(timeout = 3)
        FactorAPI.Stats.update()
        FactorAPI.Hierarchy.update()
        TradingAPI.update()
        NotificationAPI.update()

    @classmethod
    def rollback(cls , rollback_date : int):
        if not MACHINE.updatable:
            Logger.failure(f'{MACHINE.name} is not updatable, skip rollback update')
            return
        CALENDAR.check_rollback_date(rollback_date)
        DataAPI.rollback(rollback_date)
        FactorAPI.Market.rollback(rollback_date)
        FactorAPI.Stock.rollback(rollback_date , timeout = 10)
        FactorAPI.Risk.rollback(rollback_date)
        FactorAPI.Pooling.rollback(rollback_date , timeout = 10)
        FactorAPI.Stats.rollback(rollback_date)
        FactorAPI.Hierarchy.rollback(rollback_date)

    @classmethod
    def weekly(cls):
        if not MACHINE.updatable:
            Logger.failure(f'{MACHINE.name} is not updatable, skip rollback update')
            return
        ModelAPI.update_models()
        
    