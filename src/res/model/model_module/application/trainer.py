from src.proj import PATH , MACHINE , Logger , Proj
from src.proj.util import HtmlCatcher
from src.res.model.callback import CallBackManager
from src.res.model.data_module import DataModule
from src.res.model.util import BaseTrainer , RegisteredModel

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
    def initialize(cls , stage = -1 , resume = -1 , selection = -1 , base_path = None , 
                   override : dict | None = None , schedule_name = None ,
                   module = None , short_test = None , start = None , end = None ,
                   **kwargs):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        selection: [-1,choose] , [0,raw model name if resuming, create a new model name dir otherwise]  , [1,2,3,...: choose by number, start from 1]
        '''
        override = override or {}
        if module     is not None: 
            override['module'] = module
        if short_test is not None: 
            override['short_test'] = short_test
        app = cls(base_path = base_path , override = override , stage = stage , resume = resume , selection = selection , 
                  schedule_name = schedule_name , start = start , end = end , **kwargs)
        HtmlCatcher.AddExportFile(app.path_training_output)
        Proj.States.exit_files.extend(app.result_package)
        return app

    @classmethod
    def update_models(cls):
        if not MACHINE.server:
            Logger.alert1(f'{MACHINE.name} is not a server, will not update models!')
        else:
            for model in RegisteredModel.SelectModels():
                with Logger.ParagraphI(f'Updating Model {model.model_path}'):
                    cls.initialize(0 , 1 , 0 , model.model_path).go()

    @classmethod
    def train(cls , module : str | None = None , short_test : bool | None = None , 
              start : int | None = None , end : int | None = None , **kwargs):
        trainer = cls.initialize(module = module , short_test = short_test , start = start , end = end , **kwargs)
        HtmlCatcher.SetAttrs(f'Train Model of {trainer.config.model_name}')
        trainer.go()
        
        return trainer
    
    @classmethod
    def resume_train(cls , model_name : str | None = None , stage = 0 , resume = 1 , selection = 0 , 
               start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        HtmlCatcher.SetAttrs(f'Resume Model of {model_name}')
        trainer = cls.initialize(
            base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , 
            selection = selection , start = start , end = end , **kwargs)
        trainer.go()
        return trainer
    
    @classmethod
    def test(cls , model_name : str | None = None , stage = 2 , resume = 0 , selection = 0 , 
             start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        HtmlCatcher.SetAttrs(f'Test Model of {model_name}')
        trainer = cls.initialize(
            base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , 
            selection = selection, start = start , end = end , **kwargs)
        trainer.go()
        return trainer
    
    @classmethod
    def schedule(cls , schedule_name : str | None = None , short_test : bool | None = None , 
                 stage = 0 , resume = 0 , selection = 0 , start : int | None = None , end : int | None = None , **kwargs):
        assert schedule_name, 'schedule_name is required'
        HtmlCatcher.SetAttrs(f'Schedule Model of {schedule_name}')
        trainer = cls.initialize(
            schedule_name = schedule_name , short_test = short_test , stage = stage , resume = resume , 
            selection = selection , start = start , end = end , **kwargs)
        trainer.go()
        return trainer

    @classmethod
    def test_db_mapping(cls , mapping_name : str | None = None , 
                        stage = 2 , resume = 0 , selection = 0 , 
                        start : int | None = None , end : int | None = None , **kwargs):
        assert mapping_name, 'model_name is required'
        HtmlCatcher.SetAttrs(f'Test DB Mapping of {mapping_name}')
        trainer = cls.initialize(
            module = f'db@{mapping_name}' , short_test = False , stage = stage , resume = resume , 
            selection = selection, start = start , end = end , **kwargs)
        trainer.go()
        return trainer

    @classmethod
    def test_factor(cls , factor_name : str | None = None , 
                    stage = 2 , resume = 0 , selection = 0 , 
                    start : int | None = None , end : int | None = None , **kwargs):
        assert factor_name, 'factor_name is required'
        HtmlCatcher.SetAttrs(f'Test Factor of {factor_name}')
        trainer = cls.initialize(
            module = f'factor@{factor_name}' , short_test = False , stage = stage , resume = resume , 
            selection = selection, start = start , end = end , **kwargs)
        trainer.go()
        return trainer
