from src.proj import Logger , MACHINE , CALENDAR , Proj

from .data import DataAPI
from .factor import FactorAPI
from .model import ModelAPI

from .trading import TradingAPI
from .summary import SummaryAPI
from .notification import NotificationAPI

class UpdateAPI:
    @classmethod
    def daily(cls):
        """
        Orchestrate the full daily refresh: data, factors, models, trading, summaries, notifications.

        Skips early when ``MACHINE.updatable`` is false or ``DataAPI.is_updated`` fails.

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        """
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        DataAPI.update()
        if not DataAPI.is_updated():
            Logger.conclude('Data is not updated to the latest date, skip model update' , level = 'error')
            return
        FactorAPI.Market.update()
        FactorAPI.Stock.update(timeout = 3)
        FactorAPI.Affiliate.update()
        ModelAPI.update()
        FactorAPI.Pooling.update(timeout = 3)
        FactorAPI.Stats.update()
        FactorAPI.export_factor_table()
        with Proj.vb.WithVB(1):
            ModelAPI.resume_testing()
        TradingAPI.update()
        SummaryAPI.update()
        NotificationAPI.update()
        Proj.print_disk_info()

    @classmethod
    def rollback(cls , rollback_date : int):
        """
        Roll back data and factor layers to *rollback_date* (no model/trading rollback here).

        Args:
            rollback_date: Calendar-validated trade date (YYYYMMDD).

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: destructive
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        """
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        CALENDAR.check_rollback_date(rollback_date)
        DataAPI.rollback(rollback_date)
        FactorAPI.Market.rollback(rollback_date)
        FactorAPI.Stock.rollback(rollback_date , timeout = 10)
        FactorAPI.Affiliate.rollback(rollback_date)
        FactorAPI.Pooling.rollback(rollback_date , timeout = 10)
        FactorAPI.Stats.rollback(rollback_date)
        FactorAPI.export_factor_table()

    @classmethod
    def weekly(cls):
        """
        Weekly maintenance hook: currently triggers ``ModelAPI.update_models`` on updatable machines.

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        """
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        ModelAPI.update_models()

    