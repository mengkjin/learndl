from contextlib import nullcontext
from typing import Literal
from pathlib import Path
from src.proj import PATH , MACHINE , Logger , Proj  
from src.proj.util import HtmlCatcher
from src.res.model.callback import CallBackManager
from src.res.model.data_module import DataModule
from src.res.model.util import BaseTrainer , BasePredictorModel , PredictionModel , ModelPath
from src.res.factor.calculator import StockFactorHierarchy , FactorCalculator

class ModelTrainer(BaseTrainer):
    '''run through the whole process of training'''
    def init_data(self , use_data : Literal['fit','predict','both'] = 'fit' , **kwargs): 
        assert use_data != 'predict' , 'use_data cannot be predict when training models'
        self.data     = DataModule.initiate(self.config , use_data = use_data)
    def init_model(self , **kwargs):
        self.model    = BasePredictorModel.initiate(self.config , self , **kwargs)
    def init_callbacks(self , **kwargs) -> None: 
        self.callback = CallBackManager.initiate(self)

    def log_operation(self , category : str | None = None):
        self.config.model_base_path.log_operation(category)

    @classmethod
    def initialize(cls , stage = -1 , resume = -1 , selection = -1 , base_path = None , 
                   override : dict | None = None , schedule_name = None ,
                   module = None , short_test = None , start = None , end = None ,
                   use_data : Literal['fit','predict','both'] = 'fit' ,
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
                  schedule_name = schedule_name , start = start , end = end , use_data = use_data , **kwargs)
        return app

    @classmethod
    def GO(cls , *args , base_path : ModelPath | Path | str |None = None , title : str | None = None , paragraph = False , html_catcher = True, 
           check_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           log_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           use_data : Literal['fit','predict','both'] = 'fit' ,
           **kwargs):
        if title and paragraph:
            paragraph_context = Logger.Paragraph(title.title() , 1)
        else:
            paragraph_context = nullcontext()
            
        if title and html_catcher:
            html_catcher_context = HtmlCatcher(title = title.title())
        else:
            html_catcher_context = nullcontext()

        base_path = ModelPath(base_path)
        if base_path:
            last_time , time_elapsed , skip = base_path.check_last_operation(check_operation)
            if skip:
                Logger.alert1(f'{title} operated at {last_time}, {time_elapsed.total_seconds() / 3600:.1f} hours ago, will be skipped!')
                return None
            elif last_time:
                Logger.alert1(f'{title} operated at {last_time}, {time_elapsed.total_seconds() / 3600:.1f} hours ago, will run.' , vb_level = Proj.vb.max)
            else:
                Logger.stdout(f'{title} log not found, run for the first time.' , vb_level = Proj.vb.max)
                
        with paragraph_context, html_catcher_context as catcher:
            trainer = cls.initialize(*args , base_path = base_path , use_data = use_data , **kwargs)
            trainer.go().log_operation(log_operation)
            if isinstance(catcher , HtmlCatcher):
                catcher.set_export_files(trainer.html_catcher_export_path)

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
                cls.GO(0 , 1 , 0 , base_path = model.model_path , 
                       title = f'Updating Model {model.model_path}' , paragraph = True ,
                       check_operation = None if force_update else 'update_models' ,
                       log_operation = 'update_models' , use_data = 'both')

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

        for test_type , test_name in testees:
            title_object = f'{test_type.title()} {test_name}'
            cls.GO(2 , 1 , 0 , base_path = f'{test_type.lower()}@{test_name}' , short_test = False ,
                   title = f'Resume Testing {title_object}' , paragraph = True , 
                   check_operation = None if force_resume else 'resume_testing' ,
                   log_operation = 'resume_testing' , use_data = 'both')
        
    @classmethod
    def train(cls , module : str | None = None , short_test : bool | None = None , 
              start : int | None = None , end : int | None = None , **kwargs):
        return cls.GO(title = f'Train Model of Module {module}' , module = module , 
                      short_test = short_test , start = start , end = end , **kwargs)
    
    @classmethod
    def resume_train(cls , model_name : str | None = None , stage = 0 , resume = 1 , selection = 0 , 
               start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        return cls.GO(title = f'Resume Model of Model {model_name}' , 
                      base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , 
                      selection = selection , start = start , end = end , **kwargs)
    
    @classmethod
    def test(cls , model_name : str | None = None , stage = 2 , resume = 0 , selection = 0 , 
             start : int | None = None , end : int | None = None , **kwargs):
        assert model_name, 'model_name is required'
        available_models = cls.available_models(short_test = False)
        assert model_name in available_models , f'model_name {model_name} not found in {available_models}'
        return cls.GO(title = f'Test Model of Model {model_name}' , 
                      base_path = PATH.model.joinpath(model_name) , stage = stage , resume = resume , 
                      selection = selection, start = start , end = end , **kwargs)
    
    @classmethod
    def schedule(cls , schedule_name : str | None = None , short_test : bool | None = None , 
                 stage = 0 , resume = 0 , selection = 0 , start : int | None = None , end : int | None = None , **kwargs):
        assert schedule_name, 'schedule_name is required'
        return cls.GO(title = f'Schedule Model of Schedule {schedule_name}' , 
                      schedule_name = schedule_name , short_test = short_test , stage = stage , resume = resume , 
                      selection = selection , start = start , end = end , **kwargs)

    @classmethod
    def test_db_mapping(cls , mapping_name : str | None = None , 
                        stage = 2 , resume = 0 , selection = 0 , 
                        start : int | None = None , end : int | None = None , **kwargs):
        assert mapping_name, 'model_name is required'
        return cls.GO(title = f'Test DB Mapping of Mapping {mapping_name}' , 
                      module = f'db@{mapping_name}' , short_test = False , stage = stage , resume = resume , 
                      selection = selection, start = start , end = end , **kwargs)

    @classmethod
    def test_factor(cls , factor_name : str | None = None , 
                    stage = 2 , resume = 0 , selection = 0 , 
                    start : int | None = None , end : int | None = None , **kwargs):
        assert factor_name, 'factor_name is required'
        assert factor_name in cls.resumable_factors(start, end), f'factor_name {factor_name} is not available within {start} and {end}'
        return cls.GO(title = f'Test Factor of Factor {factor_name}' , 
                      module = f'factor@{factor_name}' , short_test = False , stage = stage , resume = resume , 
                      selection = selection, start = start , end = end , **kwargs)

    @classmethod
    def resumable_factors(cls , start_date : int | None = None , end_date : int | None = None , **kwargs) -> list[str]:
        factors = [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling' , updatable = True)] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside' , updatable = True)]

        available_factors = []
        for factor in factors:
            factor_dates = StockFactorHierarchy.get_factor(factor).stored_dates(start_date, end_date)
            if len(factor_dates) > 0:
                available_factors.append(factor)
        return available_factors

    @classmethod
    def resumable_models(cls , registered : bool = True , **kwargs) -> list[str]:
        available_models = cls.available_models()
        if registered:
            registered_models = [p.model_path.name for p in PredictionModel.SelectModels()]
            available_models = [p for p in available_models if p in registered_models]
        return available_models
