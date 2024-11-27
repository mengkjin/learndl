from src.model.callback import CallBackManager
from src.model.data_module import DataModule
from src.model.util import BaseTrainer
from src.basic import IS_SERVER , RegisteredModel

from ..module import get_predictor_module

class ModelTrainer(BaseTrainer):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data     = DataModule(self.config)
    def init_model(self , **kwargs):
        self.model    = get_predictor_module(self.config , **kwargs).bound_with_trainer(self)
    def init_callbacks(self , **kwargs) -> None: 
        self.callback = CallBackManager.setup(self)

    @classmethod
    def initialize(cls , stage = -1 , resume = -1 , checkname = -1 , base_path = None , override = {} , **kwargs):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        checkname: [-1,choose] , [0,default]  , [1,yes]
        '''
        app = cls(base_path = base_path , override = override , stage = stage , resume = resume , checkname = checkname , **kwargs)
        return app

    @classmethod
    def update_models(cls):
        if not IS_SERVER:
            print('This is not server! Will not update models!')
        else:
            [cls.initialize(0 , 1 , 0 , model.model_path).go() for model in RegisteredModel.SelectModels()]