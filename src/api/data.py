from src.data import DataDownloader , JsUpdater , DataUpdater , DataProcessor
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
        with EnclosedMessage(' download data '):
            DataDownloader.proceed()


        # update data from js updaters
        with EnclosedMessage(' fetch js data '):
            JsUpdater.proceed()


        # update other datas
        with EnclosedMessage(' update other datas '):
            DataUpdater.proceed()

        # update models
        with EnclosedMessage(' update factor models '):
            FactorModelUpdater.proceed()

        
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        with EnclosedMessage(' prepare predict data '):
            DataProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        with EnclosedMessage(' reconstruct historical data '):
            DataProcessor.main(predict=False)
