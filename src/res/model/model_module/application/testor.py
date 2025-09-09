from torch import Tensor

from src.res.model.data_module import DataModule , get_realistic_batch_data
from src.res.model.util import TrainConfig
from src.res.model.model_module.module import get_predictor_module

class ModelTestor:
    '''Check if a newly defined model can be forward correctly'''
    def __init__(self , module = 'tra_lstm' , data_types = 'day') -> None:
        override_cfg = {
            'env.short_test' : True ,
            'model.module' : module , 
            'model.input_type' : 'data' ,
            'model.data.types' : data_types , 
            'model.booster_head' : False
        }
        self.config = TrainConfig.default(override = override_cfg)
        self.data = DataModule(self.config , 'predict').load_data()
        self.data.setup('predict' , self.config.model_param[0] , self.data.model_date_list[0])   
        
        self.batch_data = self.data.predict_dataloader()[0]
        self.model = get_predictor_module(self.config).init_model(testor_mode = True)
        self.metrics = self.config.metrics.new_model(self.config.model_param[0])

    def __repr__(self):
        return f'{self.__class__.__name__}(model={self.model}) , check [.model][.batch_data][.metrics]'

    def try_forward(self) :
        '''as name says, try to forward'''
        if isinstance(self.batch_data.x , Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')
        self.output = self.model(self.batch_data.x)
        print(f'y shape is {self.output.pred.shape}')
        print(f'Test Forward Success')
        return self

    def try_metrics(self):
        '''as name says, try to calculate metrics'''
        if not hasattr(self , 'outputs'): self.try_forward()
        metrics = self.metrics.calculate('train' , self.batch_data.y , self.output.pred , self.batch_data.w)
        print('metrics : ' , metrics)
        print(f'Test Metrics Success')
        return self
    
    def get_realistic_batch_data(self):
        '''
        get a sample of realistic batch_data , 'day' , 'day+style' , '15m+style' ...
        day : stock_num x seq_len x 6
        30m : stock_num x seq_len x 8 x 6
        style : stock_num x 1 x 10
        indus : stock_num x 1 x 35
        ...
        '''
        return get_realistic_batch_data()
    