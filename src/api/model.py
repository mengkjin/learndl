import src.model.model_module.application as app
from src.basic import IS_SERVER
from src.data import DataProcessor
from src.func.display import EnclosedMessage

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
        with EnclosedMessage(' prepare predict data '):
            cls.prepare_predict_data()

        with EnclosedMessage(' update hidden '):
            cls.Extractor.update()

        with EnclosedMessage(' update factors '):
            cls.Predictor.update()

        with EnclosedMessage(' update factor portfolios '):
            cls.FmpBuilder.update()

    
    @classmethod
    def update_models(cls):
        '''
        Update models for both laptop and server:
        a. for laptop, do nothing
        b. for server, continue training registered models in model'
        '''
        if IS_SERVER:
            with EnclosedMessage(' reconstruct train data '):
                cls.reconstruct_train_data()

            with EnclosedMessage(' update models '):
                cls.Trainer.update_models()
        else:
            with EnclosedMessage(' update models '):
                print('This is not a server with cuda, skip this process')

    @classmethod
    def update_hidden(cls):
        '''
        Update hidden features for hidden feature models for both laptop and server:
        '''
        with EnclosedMessage(' prepare predict data '):
            cls.prepare_predict_data()

        with EnclosedMessage(' update hidden '):
            cls.Extractor.update()
    
    @classmethod
    def update_factors(cls):
        '''
        Update factors for prediction models (registered models) for both laptop and server:
        '''
        with EnclosedMessage(' prepare predict data '):
            cls.prepare_predict_data()

        with EnclosedMessage(' update factors '):
            cls.Predictor.update()
    
    @classmethod
    def test_models(cls , module = 'tra_lstm' , data_types = 'day'):
        return cls.Testor(module , data_types).try_forward()
    
    @classmethod
    def initialize_trainer(cls , stage = 0 , resume = 0 , checkname= 1):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        checkname: [-1,choose] , [0,default]  , [1,yes]
        '''
        return cls.Trainer.initialize(stage , resume , checkname)
    
    @classmethod
    def prepare_predict_data(cls): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        DataProcessor.main(predict=True)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        DataProcessor.main(predict=False)
