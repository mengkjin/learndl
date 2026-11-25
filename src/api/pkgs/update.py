"""
API for periodic update operations of this project.
"""

from __future__ import annotations
import pandas as pd
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
        factor_start = int((pd.Timestamp(str(CALENDAR.update_to())) - pd.DateOffset(years=2)).strftime('%Y%m%d'))
        FactorAPI.update_market_factors(missing_since=factor_start)
        FactorAPI.update_stock_factors(timeout = 3, missing_since=factor_start)
        FactorAPI.update_affiliate_factors(missing_since=factor_start)
        ModelAPI.update()
        FactorAPI.update_pooling_factors(timeout = 3, missing_since=factor_start)
        FactorAPI.update_factor_stats(missing_since=factor_start)
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
        Refresh training analysis inputs and extend historical fit data.

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
        from src.res.model.util.data.training_analysis_snapshot import weekly_update
        with Logger.Paragraph('Training Analysis Snapshot', 3):
            try:
                report = weekly_update()
                Logger.info(f"重算 {len(report['rewritten_years'])} 年，复用 {len(report['reused_years'])} 年。"
                            f"计算 {report['compute_seconds']:.1f} 秒，保存 {report['save_seconds']:.1f} 秒，"
                            f"构建总计 {report['total_seconds']:.1f} 秒。快照版本 {report['generation']}")
            except Exception as error:
                Logger.warning(f'Training analysis snapshot maintenance failed: {error}')
        if MACHINE.cuda_server:
            wrap_update(ModelAPI.prepare_fit_data, 'prepare fit data')
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

    