import torch

from src.proj import Logger
from src.res.model.data_module import DataModule , get_realistic_batch_data
from src.res.model.util import ModelConfig , BatchData
from src.res.model.model_module.module import get_predictor_module

class ModelTestor:
    """
    Check if a newly defined model can be forward correctly
    Example:
        testor = ModelTestor(module = 'db@scores_v0' , data_types = 'day')
        testor.try_forward()
        testor.try_metrics()
    """
    def __init__(self , module = 'tra_lstm' , data_types = 'day') -> None:
        override_cfg = {
            'env.short_test' : True ,
            'input.type' : 'data' ,
            'input.data.types' : data_types , 
            'model.module.nn.boost_head' : False
        }
        self.config = ModelConfig.default(module = module , override = override_cfg)
        self.data = DataModule(self.config , 'predict').load_data()
        self.data.setup('predict' , self.config.model_param[0] , self.data.model_date_list[0])   
        
        self.batch_input = self.data.predict_dataloader()[0]
        self.model = get_predictor_module(self.config).init_model(testor_mode = True)
        self.metrics = self.config.metrics.new_all(self.model , self.config.model_param[0])

    def __repr__(self):
        return f'{self.__class__.__name__}(model={self.model}) , check [.model][.batch_input][.metrics]'

    def try_forward(self) :
        '''as name says, try to forward'''
        if isinstance(self.batch_input.x , torch.Tensor):
            Logger.stdout(f'x shape is {self.batch_input.x.shape}')
        else:
            Logger.stdout(f'multiple x of {len(self.batch_input.x)}')
        self.output = self.model(self.batch_input.x)
        Logger.stdout(f'y shape is {self.output.pred.shape}')
        Logger.stdout(f'Test Forward Success')
        return self

    def try_metrics(self):
        '''as name says, try to calculate metrics'''
        if not hasattr(self , 'outputs'): 
            self.try_forward()
        batch_data = BatchData(self.batch_input , self.output)
        metrics = self.metrics.calculate('train' , batch_key = 'test' , batch_data = batch_data)
        Logger.stdout(f'metric output : {metrics.batch_score}')
        Logger.stdout(f'Test Metrics Success')
        return self
    
    def get_realistic_batch_data(self):
        '''
        get a sample of realistic batch_input , 'day' , 'day+style' , '15m+style' ...
        day : stock_num x seq_len x 6
        30m : stock_num x seq_len x 8 x 6
        style : stock_num x 1 x 10
        indus : stock_num x 1 x 35
        ...
        '''
        return get_realistic_batch_data()
    