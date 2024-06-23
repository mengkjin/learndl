from .module import NetDataModule
from ...data import DataProcessor , DataUpdater
from ...env import THIS_IS_SERVER

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

    @staticmethod
    def prepare_train_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        DataProcessor.main(True)
        print('-' * 80)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        assert THIS_IS_SERVER
        NetDataModule.prepare_data()
        print('-' * 80)