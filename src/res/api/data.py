from src.proj import Logger
from src.data import CoreDataUpdater , SellsideDataUpdater , AffiliatedDataUpdater , DataPreProcessor
# from src.data import JSDataUpdater
from src.res.factor.api import RiskModelUpdater , FactorCalculatorAPI
from src.basic import CALENDAR

class DataAPI:
    @staticmethod
    def wrap_update(update_func , message : str , proceed : bool = True , *args , **kwargs):
        if proceed:
            with Logger.EnclosedMessage(f' {message} '):
                update_func(*args , **kwargs)
        else:
            Logger.warning(f'Skip {message}')

    @classmethod
    def update(cls , sellside = True , risk = True , affiliated = True , factor = True): 
        """
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
        """
        # download data from tushare and other sources
        cls.wrap_update(CoreDataUpdater.update , 'download core data' , True)

        # download sellside data
        cls.wrap_update(SellsideDataUpdater.update , 'download sellside data' , sellside)

        ## update data from js updaters
        # cls.wrap_update(JSDataUpdater.update , 'fetch js data' , js)

        # update risk models
        cls.wrap_update(RiskModelUpdater.update , 'update risk models' , risk)

        # update other datas , include labels , so must be after RiskModelUpdater
        cls.wrap_update(AffiliatedDataUpdater.update , 'update affiliated datas' , affiliated)

        # update stock factor
        cls.wrap_update(FactorCalculatorAPI.update , 'update stock factors' , factor)


    @classmethod
    def update_rollback(cls , rollback_date : int , risk = True , affiliated = True , factor = True):
        """
        Rollback data to the specified date
        """
        cls.wrap_update(CoreDataUpdater.update_rollback , 'download core data' , True ,rollback_date = rollback_date)
        cls.wrap_update(RiskModelUpdater.update_rollback , 'update risk models' , risk , rollback_date = rollback_date)
        cls.wrap_update(AffiliatedDataUpdater.update_rollback , 'update affiliated datas' , affiliated , rollback_date = rollback_date)
        cls.wrap_update(FactorCalculatorAPI.update_rollback , 'update stock factors' , factor , rollback_date = rollback_date)
        
    @classmethod
    def prepare_predict_data(cls): 
        """
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        """
        cls.wrap_update(DataPreProcessor.main , 'prepare predict data' , predict = True)

    @classmethod
    def reconstruct_train_data(cls , confirm = 0): 
        """
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        """
        cls.wrap_update(DataPreProcessor.main , 'reconstruct historical data' , predict = False , confirm = confirm)
    
    @classmethod
    def is_updated(cls , verbose = False):
        """return True if the data is updated to the latest date"""
        updated = CALENDAR.updated()
        update_to = CALENDAR.update_to()
        if verbose:
            Logger.info(f'updated: {updated}, update_to: {update_to}')
        return updated >= CALENDAR.td(update_to)
