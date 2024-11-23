from src.data import DataDownloader , JsUpdater , DataUpdater , DataProcessor
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
        print('download data: ' + '*' * 20)
        DataDownloader.proceed()
        print('-' * 80)

        # update data from js updaters
        print('fetch js data: ' + '*' * 20)
        JsUpdater.proceed()
        print('-' * 80)

        # update other datas
        print('update other datas: ' + '*' * 20)
        DataUpdater.proceed()
        print('-' * 80)

        # update models
        print('update factor models: ' + '*' * 20)
        FactorModelUpdater.proceed()
        print('-' * 80)

        
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        print('prepare predict data: ' + '*' * 20)
        DataProcessor.main(predict=True)
        print('-' * 80)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        print('reconstruct historical data: ' + '*' * 20)
        DataProcessor.main(predict=False)
        print('-' * 80)