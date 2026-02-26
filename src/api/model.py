from src.res.model.util.model_path import ModelPath
import src.res.model.model_module.application as app
from src.proj import PATH , MACHINE , Logger , Proj
from src.data import DataPreProcessor

from .util import wrap_update

class ModelAPI:
    Trainer    = app.ModelTrainer
    Testor     = app.ModelTestor
    Predictor  = app.ModelPredictor
    Extractor  = app.ModelHiddenExtractor
    FmpBuilder = app.ModelPortfolioBuilder

    @classmethod
    def update(cls):
        '''
        Update prediction interims and results periodically:
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
        '''
        cls.Trainer.resume_testing(force_resume = force_resume)

    @classmethod
    def update_hidden(cls):
        '''
        Update hidden features for hidden feature models for both laptop and server:
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(cls.Extractor.update , 'update hidden')
    
    @classmethod
    def update_preds(cls):
        '''
        Update factors for prediction models for both laptop and server:
        '''
        wrap_update(cls.prepare_predict_data , 'prepare predict data')
        wrap_update(cls.Predictor.update , 'update predictors')

    @classmethod
    def recalculate_preds(cls , start_dt = None , end_dt = None):
        '''
        Recalculate factors for prediction models for both laptop and server:
        '''
        wrap_update(cls.Predictor.recalculate , 'recalculate all predictors' , start_dt = start_dt , end_dt = end_dt)
    
    @classmethod
    def test_models(cls , module = 'tra_lstm' , data_types = 'day'):
        return cls.Testor(module , data_types).try_forward()
    
    @classmethod
    def initialize_trainer(cls , stage = 0 , resume = 0 , selection = 0):
        '''
        stage:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        selection: [-1,choose] , [0,raw model name if resuming, create a new model name dir otherwise]  , [1,2,3,...: choose by number, start from 1]
        '''
        return cls.Trainer.initialize(stage = stage , resume = resume , selection = selection)
    
    @classmethod
    def prepare_predict_data(cls): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        DataPreProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        DataPreProcessor.main(predict=False , confirm=1)

    @classmethod
    def train_model(cls , module : str | None = None , short_test : bool | None = None , 
                    start : int | None = None , end : int | None = None , **kwargs):
        '''
        train a model
        '''
        with Proj.vb.WithVB(Proj.vb.max if short_test else None):
            trainer = cls.Trainer.train(module , short_test , start = start , end = end , 
                                        stage = 0 , resume = 0 , selection = 0 , **kwargs)
        return trainer

    @classmethod
    def resume_model(cls , model_name : str):
        '''
        resume a model
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
        '''
        return cls.Trainer.test(model_name , resume = int(resume) , start = start , end = end , **kwargs)

    @classmethod
    def test_factor(cls , factor_name : str | None = None , resume : int = 0 ,
                    start : int | None = None , end : int | None = None , **kwargs):
        '''
        test a existing factor
        '''
        return cls.Trainer.test_factor(factor_name , resume = resume , start = start , end = end , **kwargs)

    @classmethod
    def schedule_model(cls , schedule_name : str | None = None , short_test : bool | None = None , resume : int = 1 , 
                       start : int | None = None , end : int | None = None , **kwargs):
        '''
        Train a schedule model in config/schedule or .local_resources/shared/schedule_model folder
        '''
        return cls.Trainer.schedule(schedule_name , short_test , start = start , end = end , resume = resume , **kwargs)
    
    @classmethod
    def clear_st_models(cls):
        '''
        Clear short test models in model folder
        '''
        for path in PATH.model_st.iterdir():
            model_path = ModelPath(path)
            model_path.clear_model_path()
