"""
API for periodic update operations of this project.
"""

from __future__ import annotations
from src.proj import Logger , MACHINE , CALENDAR , Proj
from src.proj.util.filesys.ttl_cache import DiskTTLCache

from .data import DataAPI , PreProcessorTask
from .factor import FactorAPI
from .model import ModelAPI

from .trading import TradingAPI
from .summary import SummaryAPI
from .notification import NotificationAPI

from src.api.util.wrapper import wrap_update , print_update_records

__all__ = ['UpdateAPI']

class UpdateAPI:
    @classmethod
    def daily(cls):
        """
        Orchestrate the full daily refresh: data, factors, models, trading, summaries, notifications.

        Skips early when ``MACHINE.updatable`` is false or ``DataAPI.is_updated`` fails.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin, developer]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        record_entry = DiskTTLCache.get('cache_meta', 'daily_update_manual_expire')
        if not record_entry.valid_value or record_entry.valid_value < CALENDAR.update_to():
            Logger.info(f'Daily update manual set to expire due to first call to update to {CALENDAR.update_to()} ...')
            DiskTTLCache.manual_expire('daily_update')
            record_entry.put(CALENDAR.update_to())
        DataAPI.update()
        if not DataAPI.is_updated():
            Logger.conclude('Data is not updated to the latest date, skip model update' , level = 'error')
            return
        FactorAPI.update_market_factors()
        FactorAPI.update_stock_factors(timeout = 3)
        FactorAPI.update_affiliate_factors()
        ModelAPI.update()
        FactorAPI.update_pooling_factors(timeout = 3)
        FactorAPI.update_factor_stats()
        FactorAPI.export_factor_table()
        ModelAPI.resume_testing(force_resume = True , daily_update_component = True)
        TradingAPI.update()
        SummaryAPI.update()
        NotificationAPI.update()
        print_update_records()
        Proj.print_disk_info()

    @classmethod
    def rollback(cls , rollback_date : int):
        """
        Roll back data and factor layers to *rollback_date* (no model/trading rollback here).

        Args:
            rollback_date: Calendar-validated trade date (YYYYMMDD).

        [API Interaction]:
          expose: true
          email: true
          roles: [admin, developer]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        CALENDAR.check_rollback_date(rollback_date)
        DataAPI.rollback(rollback_date)
        FactorAPI.rollback_market_factors(rollback_date)
        FactorAPI.rollback_stock_factors(rollback_date , timeout = 10)
        FactorAPI.rollback_affiliate_factors(rollback_date)
        FactorAPI.rollback_pooling_factors(rollback_date , timeout = 10)
        FactorAPI.rollback_factor_stats(rollback_date)
        FactorAPI.export_factor_table()
        wrap_update(PreProcessorTask.rollback , 'rollback preprocessed data for predict' , rollback_date = rollback_date , frame = 'predict' , confirm = False)
        wrap_update(PreProcessorTask.rollback , 'rollback preprocessed data for fit' , rollback_date = rollback_date , frame = 'fit' , confirm = False)
        
        print_update_records()

    @classmethod
    def weekly(cls):
        """
        Weekly maintenance hook: currently triggers ``ModelAPI.update_models`` on updatable machines.

        [API Interaction]:
          expose: true
          email: true
          roles: [admin, developer]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        if not MACHINE.updatable:
            Logger.conclude(f'{MACHINE.name} is not updatable, skip rollback update' , level = 'error')
            return
        ModelAPI.update_models()
        print_update_records()

    @classmethod
    def tracking_port(cls):
        """
        Update tracking portfolios only.
        
        [API Interaction]:
          expose: true
          email: true
          roles: [admin, developer]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        from src.res.trading import TrackingPortfolioManager
        TrackingPortfolioManager.update()

    