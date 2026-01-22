import re
from contextlib import nullcontext
from typing import Literal
from datetime import datetime , timedelta
from src.proj import PATH , MACHINE , Logger , Proj  
from src.proj.util import HtmlCatcher
from src.res.model.callback import CallBackManager
from src.res.model.data_module import DataModule
from src.res.model.util import BaseTrainer , PredictionModel
from src.res.factor.calculator import StockFactorHierarchy , FactorCalculator

from src.res.model.model_module.module import get_predictor_module

class ModelTrainer(BaseTrainer):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data     = DataModule(self.config)
    def init_model(self , **kwargs):
        self.model    = get_predictor_module(self.config , **kwargs).bound_with_trainer(self)
    def init_callbacks(self , **kwargs) -> None: 
        self.callback = CallBackManager.setup(self)

    def log_operation(self , category : Literal['update_models' , 'resume_testing'] | None = None):
        if category is None:
            return
        else:
            path = self.config.model_base_path.log('operation_logs.log')
            for file in self.config.model_base_path.log().glob('*.log'):
                if file.stem != path.stem:
                    file.unlink()
            path.parent.mkdir(exist_ok=True)
            with open(path, 'a') as f:
                f.write(f'{category} >> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        ...

    def check_last_operation(self , category : Literal['update_models' , 'resume_testing'] | None = None) -> bool | str:
        if category is None:
            return False
        else:
            path = self.config.model_base_path.log(f'operation_logs.log')
            logs = path.read_text().split('\n') if path.exists() else []
            logs = [log for log in logs if log.startswith(category)]
            if logs:
                try:
                    time_str = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', logs[-1])[0]
                    last_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    print(last_time)
                    print(datetime.now() - timedelta(days=1))
                    if last_time > datetime.now() - timedelta(days=1):
                        return last_time.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        return False
                except (IndexError, ValueError) as e:
                    Logger.error(f'Error {e} parsing time string: {logs[-1]}')
                    return False
            else:
                return False
        ...

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
        return app

    @classmethod
    def GO(cls , *args , title : str | None = None , paragraph = False , html_catcher = True, 
           check_operation : Literal['update_models' , 'resume_testing'] | None = None , 
           log_operation : Literal['update_models' , 'resume_testing'] | None = None , **kwargs):
        if title and paragraph:
            paragraph_context = Logger.Paragraph(title.title() , 1)
        else:
            paragraph_context = nullcontext()
            
        if title and html_catcher:
            html_catcher_context = HtmlCatcher(title = title.title())
        else:
            html_catcher_context = nullcontext()

        with paragraph_context, html_catcher_context as catcher:
            trainer = cls.initialize(*args , **kwargs)
            last_operation_time = trainer.check_last_operation(check_operation)
            if last_operation_time:
                Logger.alert1(f'[{check_operation}] {title} has been done at {last_operation_time}, skip this operation!')
            else:
                trainer.go().log_operation(log_operation)
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
            for model in PredictionModel.SelectModels():
                cls.GO(0 , 1 , 0 , model.model_path , 
                       title = f'Updating Model {model.model_path}' , paragraph = True ,
                       check_operation = None if force_update else 'update_models' ,
                       log_operation = 'update_models')

        Proj.exit_files.exclude('detailed_alpha_data' , 'detailed_alpha_plot')

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

        Logger.stdout(f'Resume Testing {len(resumable_models) + len(resumable_factors)} models and factors:' , color = 'lightgreen' , bold = True)
        Logger.stdout_pairs([('Model' , model) for model in resumable_models] + [('Factor' , factor) for factor in resumable_factors] , 
                            indent = 1 , color = 'lightgreen' , bold = True)
        Logger.divider()

        for model in resumable_models:
            cls.GO(2 , 1, 0 , base_path = PATH.model.joinpath(model) , short_test = False ,
                   title = f'Resume Testing Model {model}' , paragraph = True , 
                   check_operation = None if force_resume else 'resume_testing' ,
                   log_operation = 'resume_testing')
            break

        for factor in resumable_factors:
            break
            cls.GO(2 , 1, 0 , module = f'factor@{factor}' , short_test = False ,
                    title = f'Resume Testing Factor {factor}' , paragraph = True ,
                    check_operation = None if force_resume else 'resume_testing' ,
                    log_operation = 'resume_testing')

        Proj.exit_files.exclude('detailed_alpha_data' , 'detailed_alpha_plot')

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
