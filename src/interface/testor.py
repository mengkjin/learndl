from dataclasses import dataclass
from torch import nn , Tensor

from ..util import TrainConfig , Metrics
from ..util.classes import BatchData , BatchOutput
from ..module import DataModule
from ..model import model as MODEL

@dataclass
class ModelTestor:
    '''Check if a newly defined model can be forward correctly'''
    config      : TrainConfig
    net         : nn.Module
    data_mod    : DataModule
    batch_data  : BatchData
    metrics     : Metrics

    @classmethod
    def new(cls , module = 'tra_lstm' , model_data_type = 'day'):
        config = TrainConfig.load(override = {'model_module' : module , 'model_data_type' : model_data_type} , makedir = False)
        data_mod = DataModule(config , True).load_data()
        data_mod.setup('predict' , config.model_param[0] , data_mod.model_date_list[0])   
        
        batch_data = data_mod.predict_dataloader()[0]

        net = MODEL.new(module , config.model_param[0])
        metrics = Metrics(config.train_param['criterion']).new_model(config.model_param[0] , config)
        return cls(config , net , data_mod , batch_data , metrics)

    def try_forward(self) -> None:
        '''as name says, try to forward'''
        if isinstance(self.batch_data.x , Tensor):
            print(f'x shape is {self.batch_data.x.shape}')
        else:
            print(f'multiple x of {len(self.batch_data.x)}')
        getattr(self.net , 'dynamic_data_assign' , lambda *x:None)(self)
        self.net_output = BatchOutput(self.net(self.batch_data.x))
        print(f'y shape is {self.net_output.pred.shape}')
        print(f'Test Forward Success')

    def try_metrics(self) -> None:
        '''as name says, try to calculate metrics'''
        if not hasattr(self , 'outputs'): self.try_forward()
        label , weight = self.batch_data.y , self.batch_data.w
        penalty_kwargs = {}
        penalty_kwargs.update({'net' : self.net , 'hidden' : self.net_output.hidden , 'label' : label})
        metrics = self.metrics.calculate('train' , label , self.net_output.pred , weight , self.net , penalty_kwargs)
        print('metrics : ' , metrics)
        print(f'Test Metrics Success')
    