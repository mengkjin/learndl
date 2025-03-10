from src.data import CoreDataUpdater , SellsideDataUpdater , JSDataUpdater , OtherDataUpdater , DataPreProcessor
from src.func.display import EnclosedMessage
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
        with EnclosedMessage(' download core data '):
            CoreDataUpdater.update()

        with EnclosedMessage(' download sellside data '):
            SellsideDataUpdater.update()

        # update data from js updaters
        with EnclosedMessage(' fetch js data '):
            JSDataUpdater.update()

        # update factor models
        with EnclosedMessage(' update factor models '):
            FactorModelUpdater.update()

        # update other datas
        with EnclosedMessage(' update other datas '): 
            OtherDataUpdater.update() # include labels , so must be after FactorModelUpdater

        # update stock factor
        with EnclosedMessage(' update stock factors '):
            FactorCalculatorAPI.update()
        
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        with EnclosedMessage(' prepare predict data '):
            DataPreProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(confirm = 0): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        with EnclosedMessage(' reconstruct historical data '):
            DataPreProcessor.main(predict=False , confirm = confirm)
