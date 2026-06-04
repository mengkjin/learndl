from __future__ import annotations
from contextlib import nullcontext
from typing import Literal

from src.proj import MACHINE , Proj , PATH , BaseType
from src.proj.util.io.catcher import HtmlCatcher
from src.proj.util.io.async_save import AsyncSaver
from src.res.model.util import BaseTrainer , ModelPath , PredictorPath
from src.res.factor.calculator import StockFactorHierarchy , FactorCalculator

class ModelTrainer(BaseTrainer):
    '''run through the whole process of training'''
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
    def GO(cls , * , base_path : ModelPath | BaseType.strPath | None = None , title : str | None = None , 
           paragraph : bool = False , html_catcher : bool = True, 
           check_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           log_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           use_data : Literal['fit','predict','both'] = 'fit' ,
           stage = -1 , resume = -1 , selection = -1 ,
           **kwargs):
        
        base_path = ModelPath(base_path)
        trainer = cls.initialize(
            base_path = base_path , use_data = use_data , stage = stage , 
            resume = resume , selection = selection , **kwargs)
        if base_path and check_operation:
            last_time , time_elapsed , skip = base_path.check_last_operation(check_operation)
            if skip:
                trainer.logger.skipping(f'{title} operated {time_elapsed.total_seconds() / 3600:.1f} hours ago.')
                return trainer
            elif last_time:
                trainer.logger.alert1(f'{title} operated {time_elapsed.total_seconds() / 3600:.1f} hours ago, will run.' , vb_level = 'max')
            else:
                trainer.logger.stdout(f'{title} log not found, run for the first time.' , vb_level = 'max')
    
        Paragraph = cls.logger.paragraph(title.title() , 1 , enter_vb_level = 2) if title and paragraph else nullcontext()
        Catcher = HtmlCatcher(title = title.title()) if title and html_catcher else nullcontext()

        with Paragraph, Catcher:
            trainer.go()
            base_path.log_operation(log_operation)
            if isinstance(Catcher , HtmlCatcher):
                Catcher.set_export_files(trainer.html_catcher_export_path)
            AsyncSaver.wait_all(caller_name = cls.__name__)
        return trainer

    @classmethod
    def update_models(cls , force_update : bool = False):
        if not MACHINE.cuda_server:
            cls.logger.alert1(f'{MACHINE.name} is not a server, will not update models!')
        else:
            Proj.exit_files.ban('detailed_alpha_data' , 'detailed_alpha_plot')
            for model in PredictorPath.SelectModels():
                cls.GO(base_path = model , 
                       title = f'Updating Model {model.model_name}' , paragraph = True ,
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
            cls.logger.alert1('No models or factors to resume testing!')
            return

        testees = [('Factor' , factor) for factor in resumable_factors] + [('Model' , model) for model in resumable_models]
                            
        cls.logger.stdout_pairs(testees , title = f'Resume Testing {len(resumable_models) + len(resumable_factors)} factors and models:')
        cls.logger.divider()
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
        title = f'Train Model of Module {module}' if module else 'Train Model'
        return cls.GO(title = title , module = module , 
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
    def resumable_factors(cls , start : int | None = None , end : int | None = None , **kwargs) -> list[ModelPath]:
        factors = [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling' , updatable = True)] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside' , updatable = True)]

        available_factors : list[ModelPath] = [
            ModelPath(f'factor@{factor}') for factor in factors
            if len(StockFactorHierarchy.get_factor(factor).stored_dates(start, end)) > 0
        ]
        return available_factors

    @classmethod
    def resumable_models(cls , registered : bool = True , **kwargs) -> list[ModelPath]:
        if registered:
            return [ModelPath(pred_model.full_name) for pred_model in PredictorPath.SelectModels() if pred_model.is_resumable]
        else:
            return [ModelPath(model) for model in cls.available_models()]

    @classmethod
    def all_resumable_models(cls , **kwargs) -> list[ModelPath]:
        available_models = cls.resumable_factors() + cls.resumable_models()
        return available_models

    @staticmethod
    def available_models(include_short_test : bool = False , include_factors : bool = False):
        root_paths = [PATH.model_nn , PATH.model_boost]
        if include_short_test:
            root_paths.append(PATH.model_st)
        if include_factors:
            root_paths.append(PATH.model_factor)
        bases = [f'{root.name}@{model.name}' for root in root_paths for model in root.iterdir() if model.is_dir() and not model.name.startswith('.')]
        return bases
