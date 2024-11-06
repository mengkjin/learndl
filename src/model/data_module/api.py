from typing import Optional

from .module import DataModule
from ..util import TrainConfig
from ...data import DataProcessor , DataUpdater , TushareTask
from ...basic import THIS_IS_SERVER

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
        # assert THIS_IS_SERVER
        if isinstance(data_types , str): data_types = data_types.split('+')
        DataModule.prepare_data(data_types)
        print('-' * 80)

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
        model_config = TrainConfig.load().update(short_test=True, model_data_type=model_data_type)
        data = DataModule(model_config , 'predict').load_data()
        data.setup('predict' , model_date = data.datas.y.date[-50])
        return data.predict_dataloader()[0]