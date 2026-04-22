from src.res.model.util.model_path import ModelPath
from src.res.model.model_module.application import (
  ModelTrainer , ModelTestor , ModelPredictor , ModelHiddenExtractor , ModelPortfolioBuilder)
from src.proj import PATH , MACHINE , Logger , Proj
from src.data import PreProcessorTask

from .util import wrap_update

class ModelAPI:
    @classmethod
    def update(cls):
        '''
        Update prediction interims and results periodically:

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(ModelHiddenExtractor.update , 'update hidden')
        wrap_update(ModelPredictor.update , 'update predictors')
        wrap_update(ModelPortfolioBuilder.update , 'update predictor portfolios')
    
    @classmethod
    def update_models(cls , force_update : bool = False):
        '''
        Update models for both laptop and server:
        a. for laptop, do nothing
        b. for server, continue training prediction models in model'

        Args:
            force_update: Whether to force update models even if the models are already updated recently.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: high
        '''
        if MACHINE.cuda_server:
            wrap_update(cls.reconstruct_train_data , 'reconstruct train data')
            ModelTrainer.update_models(force_update = force_update)
        else:
            Logger.alert1('This is not a server with cuda, skip this process')

    @classmethod
    def resume_testing(cls , force_resume : bool = False):
        '''
        Resume testing models for both laptop and server:

        Args:
            force_resume: Whether to force resume testing even if the models are already tested recently.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        '''
        ModelTrainer.resume_testing(force_resume = force_resume)

    @classmethod
    def update_hidden(cls):
        '''
        Update hidden features for hidden feature models for both laptop and server:

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(ModelHiddenExtractor.update , 'update hidden')
    
    @classmethod
    def update_preds(cls):
        '''
        Update factors for prediction models for both laptop and server:

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(ModelPredictor.update , 'update predictors')

    @classmethod
    def recalculate_preds(cls , start : int | None = None , end : int | None = None):
        '''
        Recalculate factors for prediction models for both laptop and server:

        Args:
            start: Start date for recalculation.
            end: End date for recalculation.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: medium
        '''
        wrap_update(ModelPredictor.recalculate , 'recalculate all predictors' , start = start , end = end)
    
    @classmethod
    def test_models(cls , module : str = 'tra_lstm' , data_types : str = 'day'):
        '''
        Lightweight forward-pass smoke test for *module* on *data_types*.

        Args:
            module: Module name.
            data_types: Data types.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 5
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        '''
        return ModelTestor(module , data_types).try_forward()
    
    @classmethod
    def initialize_trainer(cls , stage : int = 0 , resume : int = 0 , selection : int = 0):
        '''
        stage:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        selection: [-1,choose] , [0,raw model name if resuming, create a new model name dir otherwise]  , [1,2,3,...: choose by number, start from 1]
        '''
        return ModelTrainer.initialize(stage = stage , resume = resume , selection = selection)
    
    @classmethod
    def prepare_predict_data(cls): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        PreProcessorTask.update(predict=True)

    @classmethod
    def reconstruct_train_data(cls , confirm : int = 1): 
        '''
        Reconstruct historical(since 2007 , use for models starting at 2017) train data

        Args:
          confirm: Pass-through confirmation flag to ``PreProcessorTask.update`` , if zero, will prompt for confirmation.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          disable_platforms: [macos]
          execution_time: long
          memory_usage: medium
        '''
        PreProcessorTask.update(predict=False , confirm=confirm)

    @classmethod
    def train_model(cls , module : str | None = None , short_test : bool | None = None , 
                    start : int | None = None , end : int | None = None , **kwargs):
        '''
        Train a model

        Args:
          module: Module name , if None, use model specified in configs/model/default/model.yaml.
          short_test: Whether to perform a short test.
          start: Start date.
          end: End date.
          kwargs: Additional keyword arguments.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 2
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: high
        '''
        with Proj.vb.WithVB('max' if short_test else None):
            trainer = ModelTrainer.train(module , short_test , start = start , end = end , 
                                         stage = 0 , resume = 0 , selection = 0 , **kwargs)
        return trainer

    @classmethod
    def resume_model(cls , model_name : str):
        '''
        Resume a trained model

        Args:
          model_name: Model name.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: write
          lock_num: 2
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: medium
        '''
        return ModelTrainer.resume_train(model_name = model_name)
    
    @classmethod
    def test_model(cls , model_name : str | None = None , resume : int = 0 ,
                   start : int | None = None , end : int | None = None , **kwargs):
        '''
        Test a existing model

        Args:
          model_name: Model name , if None, use default model specified in configs/model/default/model.yaml.
          resume: Whether to resume testing.
          start: Start date.
          end: End date.
          kwargs: Additional keyword arguments.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 5
          disable_platforms: []
          execution_time: medium
          memory_usage: medium
        '''
        return ModelTrainer.test(model_name , resume = int(resume) , start = start , end = end , **kwargs)

    @classmethod
    def test_factor(cls , factor_name : str | None = None , resume : int = 0 ,
                    start : int | None = None , end : int | None = None , **kwargs):
        '''
        Test a existing factor

        Args:
          factor_name: Factor name , if None, use default factor specified in configs/factor/default/factor.yaml.
          resume: Whether to resume testing.
          start: Start date.
          end: End date.
          kwargs: Additional keyword arguments.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: read_only
          lock_num: -1
          disable_platforms: []
          execution_time: long
          memory_usage: high
        '''
        return ModelTrainer.test_factor(factor_name , resume = resume , start = start , end = end , **kwargs)

    @classmethod
    def schedule_model(cls , schedule_name : str | None = None , short_test : bool | None = None , resume : int = 1 , 
                       start : int | None = None , end : int | None = None , **kwargs):
        '''
        Train a schedule model in config/model/schedule or .local_resources/shared/schedule_model folder

        Args:
          schedule_name: Schedule name , if None, use default schedule specified in configs/model/schedule/schedule.yaml.
          short_test: Whether to perform a short test.
          resume: Whether to resume training.
          start: Start date.
          end: End date.
          kwargs: Additional keyword arguments.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: write
          lock_num: 2
          disable_platforms: [windows, macos]
          execution_time: long
          memory_usage: medium
        '''
        return ModelTrainer.schedule(schedule_name , short_test , start = start , end = end , resume = resume , **kwargs)
    
    @classmethod
    def clear_st_models(cls):
        '''
        Clear short test models in model folder

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: destructive
          lock_num: 1
          lock_timeout: 1
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

        Args:
          include_short_test: Whether to include short test models.
          include_factors: Whether to include factors.

        [API Interaction]:
          expose: true
          email: true
          roles: [user, developer, admin]
          risk: read_only
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: immediate
          memory_usage: low
        '''
        return ModelTrainer.available_models(include_short_test = include_short_test , include_factors = include_factors)

    @classmethod
    def rename_model(cls , old_full_name : str , new_clean_name : str):
        '''
        Rename a model

        Args:
          old_full_name: Old full name of the model.
          new_clean_name: New clean name of the model.

        [API Interaction]:
          expose: true
          email: true
          roles: [developer, admin]
          risk: write
          lock_num: 1
          lock_timeout: 1
          disable_platforms: []
          execution_time: short
          memory_usage: medium
        '''
        model_path = ModelPath(old_full_name)
        return model_path.rename(new_clean_name)
