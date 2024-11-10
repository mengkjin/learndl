from typing import Optional

from src.data import DataProcessor , DataUpdater , TushareTask , Baostock5minBarDownloader , RcquantMinBarDownloader
from src.basic import THIS_IS_SERVER
from src.model.data_module.module import DataModule

class DataAPI:
    @staticmethod
    def update(): 
        '''
        Update datas for both laptop and server:
        a. for laptop, transform data from R dataset and SQL to Database, create Updater's in './data/DataBase'
        b. for server, move Updater's to Database'
        '''
        if THIS_IS_SERVER:
            DataUpdater.update_server()
        else:
            DataUpdater.update_laptop()
        print('-' * 80)
        TushareTask.proceed()
        RcquantMinBarDownloader.proceed()
        Baostock5minBarDownloader.proceed()
        print('-' * 80)
        
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        DataProcessor.main(True)
        print('-' * 80)

    @staticmethod
    def reconstruct_train_data(data_types : Optional[list[str] | str] = None): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        DataModule.reconstruct_train_data(data_types)

    @staticmethod
    def get_realistic_batch_data(model_data_type='day'):
        '''
        get a sample of realistic batch_data , 'day' , 'day+style' , '15m+style' ...
        day : stock_num x seq_len x 6
        30m : stock_num x seq_len x 8 x 6
        style : stock_num x 1 x 10
        indus : stock_num x 1 x 35
        ...
        '''
        return DataModule.get_realistic_batch_data(model_data_type)