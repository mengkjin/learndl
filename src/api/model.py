from src.model.model_module.application import ModelTestor , ModelPredictor , ModelTrainer , ModelHiddenExtractor
from src.data import DataProcessor
from src.basic import THIS_IS_SERVER

class ModelAPI:
    Trainer = ModelTrainer
    Testor = ModelTestor
    Predictor = ModelPredictor
    HiddenExtractor = ModelHiddenExtractor

    @classmethod
    def update(cls):
        cls.prepare_predict_data()

        print('update_hidden: ' + '*' * 20)
        ModelHiddenExtractor.update_hidden()
        print('-' * 80)

        print('update_factors: ' + '*' * 20)
        ModelPredictor.update_factors()
        print('-' * 80)
    
    @classmethod
    def update_models(cls):
        '''
        Update models for both laptop and server:
        a. for laptop, do nothing
        b. for server, updated registered models in model'
        '''
        if THIS_IS_SERVER:
            cls.reconstruct_train_data()

            print('update_models: ' + '*' * 20)
            ModelTrainer.update_models()
            print('-' * 80)
        else:
            print('update_models: skip for laptop')

    @classmethod
    def update_hidden(cls):
        '''
        Update hidden features for hidden feature models for both laptop and server:
        '''
        cls.prepare_predict_data()
        
        print('update_hidden: ' + '*' * 20)
        ModelHiddenExtractor.update_hidden()
        print('-' * 80)
    
    @classmethod
    def update_factors(cls):
        '''
        Update factors for prediction models (registered models) for both laptop and server:
        '''
        cls.prepare_predict_data()

        print('update_factors: ' + '*' * 20)
        ModelPredictor.update_factors()
        print('-' * 80)
    
    @staticmethod
    def test_models(module = 'tra_lstm' , data_types = 'day'):
        return ModelTestor(module , data_types).try_forward()
    
    @staticmethod
    def initialize_trainer(stage = 0 , resume = 0 , checkname= 1):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no]       , [1,yes]
        checkname: [-1,choose] , [0,default]  , [1,yes]
        '''
        return ModelTrainer.initialize(stage , resume , checkname)
    
    @staticmethod
    def prepare_predict_data(): 
        '''
        prepare latest(1 year or so) train data for predict use, do it after 'update'
        '''
        print('prepare predict data: ' + '*' * 20)
        DataProcessor.main(predict=True)
        print('-' * 80)

    @staticmethod
    def reconstruct_train_data(): 
        '''
        reconstruct historical(since 2007 , use for models starting at 2017) train data
        '''
        print('reconstruct historical data: ' + '*' * 20)
        DataProcessor.main(predict=False)
        print('-' * 80)