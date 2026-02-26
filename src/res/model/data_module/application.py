import torch

from src.proj import Logger
from src.res.model.util import ModelConfig , BatchInput
from .module import DataModule


def get_realistic_batch_data(input_data_types='day') -> BatchInput:
    '''
    get a sample of realistic batch_input , 'day' , 'day+style' , '15m+style' ...
    day : stock_num x seq_len x 6
    30m : stock_num x seq_len x 8 x 6
    style : stock_num x 1 x 10
    indus : stock_num x 1 x 35
    ...
    '''
    override = {
        'model.module':'gru',
        'input.type':'data',
        'short_test':True,
        'input.data.types':input_data_types
    }
    model_config = ModelConfig(None, override=override, test_mode=True)
    data = DataModule(model_config , 'predict').load_data()
    data.setup('predict' , model_date = data.datas.y.date[-50])
    batch_input = data.predict_dataloader()[0]
    Logger.stdout(batch_input.info)
    return batch_input

def get_random_batch_data(dims = (30 , 6) , batch_size = 10 , predict_steps = 1) -> BatchInput:
    '''
    get a sample of random batch_input , 'day' , 'day+style' , '15m+style' ...
    '''

    x = torch.rand(batch_size , *dims)

    y = torch.rand(batch_size , predict_steps)
    w = None
    i = torch.Tensor([])
    v = torch.Tensor([])
    
    return BatchInput(x , y , w , i , v)