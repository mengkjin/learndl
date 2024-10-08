from torch import Tensor

from .module_selector import get_predictor_module
from ...data_module import DataModule
from ...util import TrainConfig

class ModelTestor:
    '''Check if a newly defined model can be forward correctly'''
    def __init__(self , module = 'tra_lstm' , data_types = 'day') -> None:
        override_cfg = {
            'model.module' : module , 
            'model.input_type' : 'data' ,
            'model.data.types' : data_types , 
            'model.booster_head' : False
        }
        self.config = TrainConfig.load(override = override_cfg , makedir = False)
        self.data = DataModule(self.config , 'predict').load_data()
        self.data.setup('predict' , self.config.model_param[0] , self.data.model_date_list[0])   
        
        self.batch_data = self.data.predict_dataloader()[0]
        self.model = get_predictor_module(self.config)
        self.metrics = self.config.metrics.new_model(self.config.model_param[0])

    def try_forward(self) -> None:
        '''as name says, try to forward'''
        if isinstance(self.batch_data.x , Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')
        self.output = self.model(self.batch_data.x)
        print(f'y shape is {self.output.pred.shape}')
        print(f'Test Forward Success')

    def try_metrics(self) -> None:
        '''as name says, try to calculate metrics'''
        if not hasattr(self , 'outputs'): self.try_forward()
        metrics = self.metrics.calculate('train' , self.batch_data.y , self.output.pred , self.batch_data.w)
        print('metrics : ' , metrics)
        print(f'Test Metrics Success')
    