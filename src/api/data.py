from src.proj import Logger
from src.data import CoreDataUpdater , SellsideDataUpdater , AffiliatedDataUpdater , DataPreProcessor
# from src.data import JSDataUpdater
from src.res.factor.api import RiskModelUpdater
from src.basic import CALENDAR

from .util import wrap_update

class DataAPI:
    @classmethod
    def update(cls , sellside = True , risk = True , affiliated = True): 
        """
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
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
        wrap_update(AffiliatedDataUpdater.update , 'update affiliated data' , skip = not affiliated)


    @classmethod
    def rollback(cls , rollback_date : int , risk = True , affiliated = True):
        """
        Rollback data to the specified date
        """
        wrap_update(CoreDataUpdater.rollback , 'rollback download core data' ,rollback_date = rollback_date)
        wrap_update(RiskModelUpdater.rollback , 'rollback risk models' , skip = not risk , rollback_date = rollback_date)
        wrap_update(AffiliatedDataUpdater.rollback , 'rollback affiliated datas' , skip = not affiliated , rollback_date = rollback_date)
        
    @classmethod
    def prepare_predict_data(cls): 
        """
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        """
        wrap_update(DataPreProcessor.main , 'prepare predict data' , predict = True)

    @classmethod
    def reconstruct_train_data(cls , confirm = 0): 
        """
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        """
        wrap_update(DataPreProcessor.main , 'reconstruct historical data' , predict = False , confirm = confirm)
    
    @classmethod
    def is_updated(cls , verbose = False):
        """return True if the data is updated to the latest date"""
        updated = CALENDAR.updated()
        update_to = CALENDAR.update_to()
        if verbose:
            Logger.info(f'updated: {updated}, update_to: {update_to}')
        return updated >= CALENDAR.td(update_to)
