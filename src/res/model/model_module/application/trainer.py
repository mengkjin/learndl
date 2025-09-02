import time

from contextlib import nullcontext

from src.proj import PATH , MACHINE , MessageCapturer
from src.basic import RegisteredModel
from src.res.model.callback import CallBackManager
from src.res.model.data_module import DataModule
from src.res.model.util import BaseTrainer

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
    def initialize(cls , stage = -1 , resume = -1 , checkname = -1 , base_path = None , 
                   override : dict | None = None , schedule_name = None ,
                   module = None , short_test = None , verbosity = None ,
                   **kwargs):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        checkname: [-1,choose] , [0,default]  , [1,yes]
        '''
        override = override or {}
        if module     is not None: override['module'] = module
        if short_test is not None: override['short_test'] = short_test
        if verbosity  is not None: override['verbosity'] = verbosity
        app = cls(base_path = base_path , override = override , stage = stage , resume = resume , checkname = checkname , schedule_name = schedule_name , **kwargs)
        return app

    @classmethod
    def update_models(cls):
        if not MACHINE.server:
            print('This is not server! Will not update models!')
        else:
            for model in RegisteredModel.SelectModels():
                print(f'Updating model: {model.model_path}')
                print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
                cls.initialize(0 , 1 , 0 , model.model_path).go()
                print(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    @classmethod
    def train(cls , module : str | None = None , short_test : bool | None = None , message_capturer : bool = True , **kwargs):
        with MessageCapturer.CreateCapturer(message_capturer) as capturer:
            trainer = cls.initialize(module = module , short_test = short_test , **kwargs).go()
            capturer.set_attrs(f'Train Model of {trainer.config.model_name}' , trainer.path_training_output)
        return trainer
    
    @classmethod
    def resume(cls , model_name : str | None = None , message_capturer : bool = True , 
               stage = 0 , resume = 1 , checkname = 1 , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        with MessageCapturer.CreateCapturer(message_capturer) as capturer:
            trainer = cls.initialize(base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , checkname = checkname , **kwargs).go()
            capturer.set_attrs(f'Resume Model of {trainer.config.model_name}' , trainer.path_training_output)
        return trainer
    
    @classmethod
    def test(cls , model_name : str | None = None , short_test : bool | None = None , message_capturer : bool = True , 
             stage = 2 , resume = 1 , checkname = 1 , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        with MessageCapturer.CreateCapturer(message_capturer) as capturer:
            trainer = cls.initialize(base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , checkname = checkname , **kwargs).go()
            capturer.set_attrs(f'Test Model of {trainer.config.model_name}' , trainer.path_training_output)
        return trainer
    
    @classmethod
    def schedule(cls , schedule_name : str | None = None , short_test : bool | None = None , message_capturer : bool = True , **kwargs):
        assert schedule_name, 'schedule_name is required'
        with MessageCapturer.CreateCapturer(message_capturer) as capturer:
            trainer = cls.initialize(schedule_name = schedule_name , short_test = short_test , **kwargs).go()
            capturer.set_attrs(f'Schedule Model of {trainer.config.model_name}' , trainer.path_training_output)
        return trainer
