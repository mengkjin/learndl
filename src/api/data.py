from src.basic import THIS_IS_SERVER
from src.data import (DataProcessor , DataUpdater , TushareDownloader , OtherSourceDownloader ,
                      ClassicLabelsUpdater)
from src.factor.calculator import FactorModelUpdater

class DataAPI:
    @staticmethod
    def update(): 
        '''
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
        '''
        print('update data: ' + '*' * 20)
        if THIS_IS_SERVER:
            DataUpdater.update_server()
        else:
            DataUpdater.update_laptop()
        print('-' * 80)
        TushareDownloader.proceed()
        OtherSourceDownloader.proceed()
        print('-' * 80)
        FactorModelUpdater.proceed()
        print('-' * 80)
        ClassicLabelsUpdater.proceed()
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