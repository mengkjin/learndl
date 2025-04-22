import torch

from typing import Optional

from src.model.util import TrainConfig , BatchData
from .module import DataModule


def get_realistic_batch_data(model_data_type='day') -> BatchData:
    '''
    get a sample of realistic batch_data , 'day' , 'day+style' , '15m+style' ...
    day : stock_num x seq_len x 6
    30m : stock_num x seq_len x 8 x 6
    style : stock_num x 1 x 10
    indus : stock_num x 1 x 35
    ...
    '''
    model_config = TrainConfig.load().update(short_test=True, model_data_type=model_data_type)
    data = DataModule(model_config , 'predict').load_data()
    data.setup('predict' , model_date = data.datas.y.date[-50])
    return data.predict_dataloader()[0]

def get_random_batch_data(dims = (30 , 6) , batch_size = 10 , predict_steps = 1) -> BatchData:
    '''
    get a sample of random batch_data , 'day' , 'day+style' , '15m+style' ...
    '''

    x = torch.rand(batch_size , *dims)

    y = torch.rand(batch_size , predict_steps)
    w = None
    i = torch.Tensor([])
    v = torch.Tensor([])
    
    return BatchData(x , y , w , i , v)