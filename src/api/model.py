from src.res.model.util.model_path import ModelPath
import src.res.model.model_module.application as app
from src.proj import PATH , MACHINE , Logger , Proj
from src.data import PreProcessorTask

from .util import wrap_update

class ModelAPI:
    Trainer    = app.ModelTrainer
    Testor     = app.ModelTestor
    Predictor  = app.ModelPredictor
    Extractor  = app.ModelHiddenExtractor
    FmpBuilder = app.ModelPortfolioBuilder
    Calculator = app.ModelCalculator

    @classmethod
    def update(cls):
        '''
        Update prediction interims and results periodically:

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(cls.Extractor.update , 'update hidden')
        wrap_update(cls.Predictor.update , 'update predictors')
        wrap_update(cls.FmpBuilder.update , 'update predictor portfolios')
    
    @classmethod
    def update_models(cls , force_update = False):
        '''
        Update models for both laptop and server:
        a. for laptop, do nothing
        b. for server, continue training prediction models in model'

        Args:
            force_update: Passed to ``Trainer.update_models`` on CUDA servers.

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: write
          lock_num: -1
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: high
        '''
        if MACHINE.cuda_server:
            wrap_update(cls.reconstruct_train_data , 'reconstruct train data')
            cls.Trainer.update_models(force_update = force_update)
        else:
            Logger.alert1('This is not a server with cuda, skip this process')

    @classmethod
    def resume_testing(cls , force_resume = False):
        '''
        Resume testing models for both laptop and server:

        Args:
            force_resume: Forwarded to ``Trainer.resume_testing``.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        cls.Trainer.resume_testing(force_resume = force_resume)

    @classmethod
    def update_hidden(cls):
        '''
        Update hidden features for hidden feature models for both laptop and server:

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(cls.Extractor.update , 'update hidden')
    
    @classmethod
    def update_preds(cls):
        '''
        Update factors for prediction models for both laptop and server:

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(cls.Predictor.update , 'update predictors')

    @classmethod
    def recalculate_preds(cls , start = None , end = None):
        '''
        Recalculate factors for prediction models for both laptop and server:

        Args:
            start, end: Optional date window for recalculation.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        wrap_update(cls.Predictor.recalculate , 'recalculate all predictors' , start = start , end = end)
    
    @classmethod
    def test_models(cls , module = 'tra_lstm' , data_types = 'day'):
        '''
        Lightweight forward-pass smoke test for *module* on *data_types*.

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        return cls.Testor(module , data_types).try_forward()
    
    @classmethod
    def initialize_trainer(cls , stage = 0 , resume = 0 , selection = 0):
        '''
        stage:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        selection: [-1,choose] , [0,raw model name if resuming, create a new model name dir otherwise]  , [1,2,3,...: choose by number, start from 1]

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        '''
        return cls.Trainer.initialize(stage = stage , resume = resume , selection = selection)
    
    @classmethod
    def prepare_predict_data(cls): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: medium
          memory_usage: high
        '''
        PreProcessorTask.update(predict=True)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        PreProcessorTask.update(predict=False , confirm=1)

    @classmethod
    def train_model(cls , module : str | None = None , short_test : bool | None = None , 
                    start : int | None = None , end : int | None = None , **kwargs):
        '''
        train a model

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: high
        '''
        with Proj.vb.WithVB('max' if short_test else None):
            trainer = cls.Trainer.train(module , short_test , start = start , end = end , 
                                        stage = 0 , resume = 0 , selection = 0 , **kwargs)
        return trainer

    @classmethod
    def resume_model(cls , model_name : str):
        '''
        resume a model

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: high
        '''
        return cls.Trainer.resume_train(model_name = model_name)
    
    @classmethod
    def test_model(cls , model_name : str | None = None , resume : int = 0 ,
                   start : int | None = None , end : int | None = None , **kwargs):
        '''
        test a existing model
        model_name :
            None: use default model
            str : use the model name , must be in ModelAPI.Trainer.available_models(short_test = False)

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        return cls.Trainer.test(model_name , resume = int(resume) , start = start , end = end , **kwargs)

    @classmethod
    def test_factor(cls , factor_name : str | None = None , resume : int = 0 ,
                    start : int | None = None , end : int | None = None , **kwargs):
        '''
        test a existing factor

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        return cls.Trainer.test_factor(factor_name , resume = resume , start = start , end = end , **kwargs)

    @classmethod
    def schedule_model(cls , schedule_name : str | None = None , short_test : bool | None = None , resume : int = 1 , 
                       start : int | None = None , end : int | None = None , **kwargs):
        '''
        Train a schedule model in config/model/schedule or .local_resources/shared/schedule_model folder

        [API Interaction]:
          expose: false
          roles: [developer, admin]
          risk: write
          lock_num: -1
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: high
        '''
        return cls.Trainer.schedule(schedule_name , short_test , start = start , end = end , resume = resume , **kwargs)
    
    @classmethod
    def clear_st_models(cls):
        '''
        Clear short test models in model folder

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: destructive
          lock_num: -1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        for path in PATH.model_st.iterdir():
            model_path = ModelPath(path)
            model_path.clear_model_path()

    @classmethod
    def available_models(cls , include_short_test : bool = False , include_factors : bool = False):
        '''
        Get available models in model folder

        [API Interaction]:
          expose: false
          roles: [user, developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        '''
        return cls.Trainer.available_models(include_short_test = include_short_test , include_factors = include_factors)

    @classmethod
    def rename_model(cls , old_full_name : str , new_clean_name : str):
        '''
        Rename a model

        [API Interaction]:
          expose: false
          roles: [admin]
          risk: write
          lock_num: -1
          disable_platforms: []
          execution_time: short
          memory_usage: low
        '''
        model_path = ModelPath(old_full_name)
        return model_path.rename(new_clean_name)
