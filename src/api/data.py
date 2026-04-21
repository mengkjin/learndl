from src.proj import Logger , CALENDAR
from src.data import CoreDataUpdater , SellsideDataUpdater , CustomDataUpdater , PreProcessorTask
from src.data.crawler import AnnouncementAgent
# from src.data import JSDataUpdater
from src.res.factor.api import RiskModelUpdater

from .util import wrap_update

class DataAPI:
    @classmethod
    def update(cls , sellside : bool = True , risk : bool = True , affiliated : bool = True): 
        """
        Update core, sellside, risk, custom, and announcement data pipelines for laptop and server roles.

        Args:
            sellside: When false, skip sellside download/update.
            risk: When false, skip risk-model update.
            affiliated: When false, skip custom/affiliated data update.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        # download data from tushare and other sources
        wrap_update(CoreDataUpdater.update , 'download core data')

        # download sellside data
        wrap_update(SellsideDataUpdater.update , 'download sellside data' , skip = not sellside)

        ## update data from js updaters
        # cls.wrap_update(JSDataUpdater.update , 'fetch js data' , js)

        # update risk models
        wrap_update(RiskModelUpdater.update , 'update risk models' , skip = not risk)

        # update other datas , include labels , so must be after RiskModelUpdater
        wrap_update(CustomDataUpdater.update , 'update affiliated data' , skip = not affiliated)

        # update announcement data
        wrap_update(AnnouncementAgent.update , 'update announcement data')


    @classmethod
    def rollback(cls , rollback_date : int , risk : bool = True , affiliated : bool = True):
        """
        Rollback core, risk, and affiliated data to *rollback_date* (calendar-validated by callers).

        Args:
            rollback_date: Trade date (YYYYMMDD) to roll back toward.
            risk: When false, skip risk-model rollback.
            affiliated: When false, skip custom/affiliated rollback.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        wrap_update(CoreDataUpdater.rollback , 'rollback download core data' ,rollback_date = rollback_date)
        wrap_update(RiskModelUpdater.rollback , 'rollback risk models' , skip = not risk , rollback_date = rollback_date)
        wrap_update(CustomDataUpdater.rollback , 'rollback affiliated datas' , skip = not affiliated , rollback_date = rollback_date)
        
    @classmethod
    def prepare_predict_data(cls): 
        """
        Build or refresh the latest prediction-oriented train window (typically ~1y) via ``PreProcessorTask``.
        Run after routine ``update`` when prediction pipelines need fresh inputs.

        [API Interaction]:
          expose: true
          email: false
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: high
        """
        wrap_update(PreProcessorTask.update , 'prepare predict data' , predict = True)

    @classmethod
    def reconstruct_train_data(cls , confirm : int = 0): 
        """
        Rebuild long-history training tables (since ~2007) for models that start around 2017.

        Args:
            confirm: Pass-through confirmation flag to ``PreProcessorTask.update`` , if zero, will prompt for confirmation.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: high
        """
        wrap_update(PreProcessorTask.update , 'reconstruct historical data' , predict = False , confirm = confirm)
    
    @classmethod
    def is_updated(cls):
        """
        Whether the dataset is updated through the latest expected trade date
        
        Args:
          No arguments
        Returns:
          bool: True iff local updated date is not before the calendar's required update-to trade date

        [API Interaction]:
          expose: true
          email: false
          roles: [user, developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        """
        updated = CALENDAR.updated()
        update_to = CALENDAR.td(CALENDAR.update_to()) # use trade date to compare
        is_updated = updated >= update_to
        if not is_updated:
            Logger.alert1(f'Only updated to {updated}, should update to {update_to}')
        return is_updated
