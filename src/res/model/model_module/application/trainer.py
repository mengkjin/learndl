from contextlib import nullcontext
from typing import Literal
from pathlib import Path
from src.proj import MACHINE , Logger , Proj  
from src.proj.util import HtmlCatcher
from src.res.model.callback import CallBackManager
from src.res.model.data_module import DataModule
from src.res.model.util import BaseTrainer , BasePredictorModel , PredictionModel , ModelPath , ModelConfig
from src.res.factor.calculator import StockFactorHierarchy , FactorCalculator

class ModelTrainer(BaseTrainer):
    '''run through the whole process of training'''
    def init_config(self , base_path = None , * , module : str | None = None , schedule_name = None , override : dict | None = None , **kwargs) -> None:
        '''initialized configuration'''
        self.config   = ModelConfig.initiate(base_path , module = module , schedule_name = schedule_name , override = override , min_key_len = 30 , **kwargs)
    def init_data(self , use_data : Literal['fit','predict','both'] = 'fit' , **kwargs): 
        assert use_data != 'predict' , 'use_data cannot be predict when training models'
        self.data     = DataModule.initiate(self.config , use_data = use_data , min_key_len = 30)
    def init_model(self , **kwargs):
        self.model    = BasePredictorModel.initiate(self.config , self , min_key_len = 30 , **kwargs)
    def init_callbacks(self , **kwargs) -> None: 
        self.callback = CallBackManager.initiate(self , min_key_len = 30)

    @classmethod
    def initialize(cls , base_path = None , * ,
                   module = None , schedule_name = None , override : dict | None = None , 
                   short_test = None , start = None , end = None ,
                   use_data : Literal['fit','predict','both'] = 'fit' ,
                   stage = -1 , resume = -1 , selection = -1 , 
                   **kwargs):
        '''
        state:     [-1,choose if optional] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose if optional] , [0,no]       , [1,yes]
        selection: [-1,choose if optional] , [0,raw model name unless fitting and not resuming]  , [1,2,3,...: choose by number, start from 1]
        '''
        app = cls(base_path = base_path , 
                  module = module , schedule_name = schedule_name , override = override ,
                  stage = stage , resume = resume , selection = selection , 
                  start = start , end = end , use_data = use_data , short_test = short_test , **kwargs)
        return app
        
    @classmethod
    def GO(cls , * , base_path : ModelPath | Path | str | None = None , title : str | None = None , 
           paragraph : bool = False , html_catcher : bool = True, 
           check_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           log_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           use_data : Literal['fit','predict','both'] = 'fit' ,
           stage = -1 , resume = -1 , selection = -1 ,
           **kwargs):
        if title and paragraph:
            paragraph_context = Logger.Paragraph(title.title() , 1)
        else:
            paragraph_context = nullcontext()
            
        if title and html_catcher:
            html_catcher_context = HtmlCatcher(title = title.title())
        else:
            html_catcher_context = nullcontext()
       
        with paragraph_context, html_catcher_context as catcher:
            base_path = ModelPath(base_path)
            trainer = cls.initialize(base_path = base_path , use_data = use_data , 
                                     stage = stage , resume = resume , selection = selection , **kwargs)
            
            if base_path:
                last_time , time_elapsed , skip = base_path.check_last_operation(check_operation)
                if skip:
                    Logger.alert1(f'{title} operated at {last_time}, {time_elapsed.total_seconds() / 3600:.1f} hours ago, will be skipped!')
                    return trainer
                elif last_time:
                    Logger.alert1(f'{title} operated at {last_time}, {time_elapsed.total_seconds() / 3600:.1f} hours ago, will run.' , vb_level = Proj.vb.max)
                else:
                    Logger.stdout(f'{title} log not found, run for the first time.' , vb_level = Proj.vb.max)
         
            trainer.go()
            base_path.log_operation(log_operation)
            if isinstance(catcher , HtmlCatcher):
                catcher.set_export_files(trainer.html_catcher_export_path)

        return trainer

    @classmethod
    def update_models(cls , force_update = False):
        # if not MACHINE.server:
        #     Logger.alert1(f'{MACHINE.name} is not a server, will not update models!')
        # else:
        #     for model in PredictionModel.SelectModels():
        #         with Logger.ParagraphI(f'Updating Model {model.model_path}'):
        #             cls.initialize(0 , 1 , 0 , model.model_path).go()
        if not MACHINE.cuda_server:
            Logger.alert1(f'{MACHINE.name} is not a server, will not update models!')
        else:
            Proj.exit_files.ban('detailed_alpha_data' , 'detailed_alpha_plot')
            for model in PredictionModel.SelectModels():
                cls.GO(base_path = model.model_path , 
                       title = f'Updating Model {model.model_path.model_name}' , paragraph = True ,
                       check_operation = None if force_update else 'update_models' ,
                       log_operation = 'update_models' , use_data = 'both' ,
                       stage = 0 , resume = 1 , selection = 0)

    @classmethod
    def resume_testing(cls , models = True , factors = True , force_resume = False):
        '''
        Resume testing prediction models and factors:
        '''

        resumable_models = cls.resumable_models() if models else []
        resumable_factors = cls.resumable_factors() if factors else []

        if len(resumable_models) + len(resumable_factors) == 0:
            Logger.alert1('No models or factors to resume testing!')
            return

        testees = [('Factor' , factor) for factor in resumable_factors] + [('Model' , model) for model in resumable_models]
                            
        Logger.stdout_pairs(testees , title = f'Resume Testing {len(resumable_models) + len(resumable_factors)} factors and models:')
        Logger.divider()
        Proj.exit_files.ban('detailed_alpha_data' , 'detailed_alpha_plot')

        for testee_type , testee_path in testees:
            title_object = f'{testee_type.title()} {testee_path.base.name}'
            cls.GO(base_path = testee_path , short_test = False ,
                   title = f'Resume Testing {title_object}' , paragraph = True , 
                   check_operation = None if force_resume else 'resume_testing' ,
                   log_operation = 'resume_testing' , use_data = 'both' ,
                   stage = 2 , resume = 1 , selection = 0)
        
    @classmethod
    def train(cls , module : str | None = None , short_test : bool | None = None , 
              start : int | None = None , end : int | None = None , **kwargs):
        return cls.GO(title = f'Train Model of Module {module}' , module = module , 
                      short_test = short_test , start = start , end = end , **kwargs)
    
    @classmethod
    def resume_train(cls , model_name : str | None = None , stage = 0 , resume = 1 , selection = 0 , 
               start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        base_path = ModelPath(model_name)
        assert base_path.base.exists() , f'model_name {model_name} not found'
        return cls.GO(title = f'Resume Model of Model {model_name}' , 
                      base_path = base_path , stage = stage , resume = resume , 
                      selection = selection , start = start , end = end , **kwargs)
    
    @classmethod
    def test(cls , model_name : str | None = None , stage = 2 , resume = 0 , selection = 0 , 
             start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        base_path = ModelPath(model_name)
        assert base_path.base.exists() , f'model_name {model_name} not found'
        return cls.GO(title = f'Test Model of Model {model_name}' , 
                      base_path = base_path , stage = stage , resume = resume , 
                      selection = selection, start = start , end = end , **kwargs)
    
    @classmethod
    def schedule(cls , schedule_name : str | None = None , short_test : bool | None = None , 
                 stage = 0 , resume = 0 , selection = 0 , start : int | None = None , end : int | None = None , **kwargs):
        assert schedule_name, 'schedule_name is required'
        return cls.GO(title = f'Schedule Model of Schedule {schedule_name}' , 
                      schedule_name = schedule_name , short_test = short_test , stage = stage , resume = resume , 
                      selection = selection , start = start , end = end , **kwargs)

    @classmethod
    def test_factor(cls , factor_name : str | None = None , 
                    stage = 2 , resume = 0 , selection = 0 , 
                    start : int | None = None , end : int | None = None , **kwargs):
        assert factor_name, 'factor_name is required'
        base_path = ModelPath(f'factor@{factor_name}')
        assert base_path in cls.resumable_factors(start, end), f'factor_name {factor_name} is not available within {start} and {end}'
        return cls.GO(title = f'Test Factor of Factor {factor_name}' , 
                      base_path = base_path , short_test = False , stage = stage , resume = resume , 
                      selection = selection, start = start , end = end , **kwargs)

    @classmethod
    def resumable_factors(cls , start_date : int | None = None , end_date : int | None = None , **kwargs) -> list[ModelPath]:
        factors = [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling' , updatable = True)] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside' , updatable = True)]

        available_factors : list[ModelPath] = [
            ModelPath(f'factor@{factor}') for factor in factors
            if len(StockFactorHierarchy.get_factor(factor).stored_dates(start_date, end_date)) > 0
        ]
        return available_factors

    @classmethod
    def resumable_models(cls , registered : bool = True , **kwargs) -> list[ModelPath]:
        if registered:
            return [pred_model.model_path for pred_model in PredictionModel.SelectModels() if pred_model.model_path.base.exists()]
        else:
            return [ModelPath(model) for model in cls.available_models()]

    @classmethod
    def all_resumable_models(cls , **kwargs) -> list[ModelPath]:
        available_models = cls.resumable_factors() + cls.resumable_models()
        return available_models
