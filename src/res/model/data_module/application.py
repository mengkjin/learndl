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
    model_config = ModelConfig(override=override)
    data = DataModule(model_config , 'predict').load_data()
    data.setup('predict' , model_date = data.datas.y.date[-50])
    batch_input = data.predict_dataloader()[0]
    Logger.stdout(batch_input.info)
    return batch_input

def get_random_batch_data(batch_size = 10 , seq_len = 30 , n_inputs = 6 , predict_steps = 1) -> BatchInput:
    '''
    get a sample of random batch_input , 'day' , 'day+style' , '15m+style' ...
    '''
    return BatchInput.random(batch_size , seq_len , n_inputs , predict_steps)
