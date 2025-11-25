import time

from src.proj import PATH , MACHINE , HtmlCatcher
from src.basic import RegisteredModel , Email
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
                   start = None , end = None ,
                   **kwargs):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        checkname: [-1,choose] , [0,default]  , [1,yes]
        '''
        override = override or {}
        if module     is not None: 
            override['module'] = module
        if short_test is not None: 
            override['short_test'] = short_test
        if verbosity  is not None: 
            override['verbosity'] = verbosity
        app = cls(base_path = base_path , override = override , stage = stage , resume = resume , checkname = checkname , 
                  schedule_name = schedule_name , start = start , end = end , **kwargs)
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
    def train(cls , module : str | None = None , short_test : bool | None = None , 
              start : int | None = None , end : int | None = None , **kwargs):
        with HtmlCatcher(True) as catcher:
            trainer = cls.initialize(module = module , short_test = short_test , start = start , end = end , **kwargs)
            trainer.go()
            catcher.set_attrs(f'Train Model of {trainer.config.model_name}' , trainer.path_training_output)
        Email.Attach(trainer.result_package)
        return trainer
    
    @classmethod
    def resume(cls , model_name : str | None = None , stage = 0 , resume = 1 , checkname = -1 , 
               start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        with HtmlCatcher(True) as catcher:
            trainer = cls.initialize(
                base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , 
                checkname = checkname , start = start , end = end , **kwargs)
            trainer.go()
            catcher.set_attrs(f'Resume Model of {trainer.config.model_name}' , trainer.path_training_output)
        Email.Attach(trainer.result_package)
        return trainer
    
    @classmethod
    def test(cls , model_name : str | None = None , short_test : bool | None = None , stage = 2 , resume = 1 , checkname = -1 , 
             start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        with HtmlCatcher(True) as catcher:
            trainer = cls.initialize(
                base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , 
                checkname = checkname, start = start , end = end , **kwargs)
            trainer.go()
            catcher.set_attrs(f'Test Model of {trainer.config.model_name}' , trainer.path_training_output)
        Email.Attach(trainer.result_package)
        return trainer
    
    @classmethod
    def schedule(cls , schedule_name : str | None = None , short_test : bool | None = None , 
                 stage = 0 , resume = 0 , checkname = -1 , start : int | None = None , end : int | None = None , **kwargs):
        assert schedule_name, 'schedule_name is required'
        with HtmlCatcher(True) as catcher:
            trainer = cls.initialize(
                schedule_name = schedule_name , short_test = short_test , stage = stage , resume = resume , 
                checkname = checkname , start = start , end = end , **kwargs)
            trainer.go()
            catcher.set_attrs(f'Schedule Model of {trainer.config.model_name}' , trainer.path_training_output)
        Email.Attach(trainer.result_package)
        return trainer

    @classmethod
    def test_db_mapping(cls , mapping_name : str | None = None , short_test : bool | None = None , 
                        stage = 2 , resume = 0 , checkname = -1 , start : int | None = None , end : int | None = None , **kwargs):
        assert mapping_name, 'model_name is required'
        with HtmlCatcher(True) as catcher:
            trainer = cls.initialize(
                module = f'db@{mapping_name}' , short_test = short_test , stage = stage , resume = resume , 
                checkname = checkname, start = start , end = end , **kwargs)
            trainer.go()
            catcher.set_attrs(f'Test DB Mapping of {mapping_name}' , trainer.path_training_output)
        Email.Attach(trainer.result_package)
        return trainer

    @classmethod
    def test_factor(cls , factor_name : str | None = None , short_test : bool | None = None , 
                    stage = 2 , resume = 0 , checkname = -1 , start : int | None = None , end : int | None = None , **kwargs):
        assert factor_name, 'factor_name is required'
        with HtmlCatcher(True) as catcher:
            trainer = cls.initialize(
                module = f'factor@{factor_name}' , short_test = short_test , stage = stage , resume = resume , 
                checkname = checkname, start = start , end = end , **kwargs)
            trainer.go()
            catcher.set_attrs(f'Test Factor of {factor_name}' , trainer.path_training_output)
        Email.Attach(trainer.result_package)
        return trainer
