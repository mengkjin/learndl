from src.data import CoreDataUpdater , SellsideDataUpdater , JSDataUpdater , OtherDataUpdater , DataProcessor
from src.func.display import EnclosedMessage
from src.factor.calculator import FactorModelUpdater

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

        # update other datas
        with EnclosedMessage(' update other datas '):
            OtherDataUpdater.update()

        # update models
        with EnclosedMessage(' update factor models '):
            FactorModelUpdater.update()

        
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        with EnclosedMessage(' prepare predict data '):
            DataProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(confirm = 0): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        with EnclosedMessage(' reconstruct historical data '):
            DataProcessor.main(predict=False , confirm = confirm)
