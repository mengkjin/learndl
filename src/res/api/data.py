from src.proj import Logger
from src.data import CoreDataUpdater , SellsideDataUpdater , OtherDataUpdater , DataPreProcessor , ModuleData
# from src.data import JSDataUpdater
from src.res.factor.api import FactorModelUpdater , FactorCalculatorAPI
from src.basic import CALENDAR

class DataAPI:
    @staticmethod
    def update(): 
        """
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
        """
        # download data from tushare and other sources
        with Logger.EnclosedMessage(' download core data '):
            CoreDataUpdater.update()

        with Logger.EnclosedMessage(' download sellside data '):
            SellsideDataUpdater.update()

        ## update data from js updaters
        #with Logger.EnclosedMessage(' fetch js data '):
        #    JSDataUpdater.update()

        # update factor models
        with Logger.EnclosedMessage(' update factor models '):
            FactorModelUpdater.update()

        # update other datas
        with Logger.EnclosedMessage(' update other datas '): 
            OtherDataUpdater.update() # include labels , so must be after FactorModelUpdater

        # update stock factor
        with Logger.EnclosedMessage(' update stock factors '):
            FactorCalculatorAPI.update()

        with Logger.EnclosedMessage(' purge old data cache '):
            ModuleData.purge_all()


    @staticmethod
    def update_rollback(rollback_date : int):
        """
        Rollback data to the specified date
        """
        with Logger.EnclosedMessage(f' download core data , rollback to {rollback_date}'):
            CoreDataUpdater.update_rollback(rollback_date)
        with Logger.EnclosedMessage(f' update other datas , rollback to {rollback_date}'): 
            OtherDataUpdater.update_rollback(rollback_date)
        with Logger.EnclosedMessage(f' update factor models , rollback to {rollback_date}'):
            FactorModelUpdater.update_rollback(rollback_date)
        with Logger.EnclosedMessage(f' update stock factors , rollback to {rollback_date}'):
            FactorCalculatorAPI.update_rollback(rollback_date)
        
    @staticmethod
    def prepare_predict_data(): 
        """
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        """
        with Logger.EnclosedMessage(' prepare predict data '):
            DataPreProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(confirm = 0): 
        """
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        """
        with Logger.EnclosedMessage(' reconstruct historical data '):
            DataPreProcessor.main(predict=False , confirm = confirm)

    @staticmethod
    def is_updated(verbose = False):
        """return True if the data is updated to the latest date"""
        updated = CALENDAR.updated()
        update_to = CALENDAR.update_to()
        if verbose:
            Logger.info(f'updated: {updated}, update_to: {update_to}')
        return updated >= CALENDAR.td(update_to)
