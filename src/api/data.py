from src.data import CoreDataUpdater , SellsideDataUpdater , JSDataUpdater , OtherDataUpdater , DataPreProcessor
from src.basic import Logger
from src.factor.api import FactorModelUpdater , FactorCalculatorAPI

class DataAPI:
    @staticmethod
    def update(): 
        '''
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
        '''
        # download data from tushare and other sources
        with Logger.EnclosedMessage(' download core data '):
            CoreDataUpdater.update()

        with Logger.EnclosedMessage(' download sellside data '):
            SellsideDataUpdater.update()

        # update data from js updaters
        with Logger.EnclosedMessage(' fetch js data '):
            JSDataUpdater.update()

        # update factor models
        with Logger.EnclosedMessage(' update factor models '):
            FactorModelUpdater.update()

        # update other datas
        with Logger.EnclosedMessage(' update other datas '): 
            OtherDataUpdater.update() # include labels , so must be after FactorModelUpdater

        # update stock factor
        with Logger.EnclosedMessage(' update stock factors '):
            FactorCalculatorAPI.update()
        
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        with Logger.EnclosedMessage(' prepare predict data '):
            DataPreProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(confirm = 0): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        with Logger.EnclosedMessage(' reconstruct historical data '):
            DataPreProcessor.main(predict=False , confirm = confirm)
