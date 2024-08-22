import numpy as np

from typing import Any , Literal

from ...util import BaseDataModule , BaseTrainer , BatchData , BatchOutput , BoosterInput , get_booster_type , GeneralBooster

class BoosterModel:
    '''Booster used in nn_model'''
    def __init__(self , model_module : BaseTrainer) -> None:
        self.module = model_module
        self.model : GeneralBooster

    def __bool__(self): return True
    @property
    def data(self) -> BaseDataModule: return self.module.data
    @property
    def train_dl(self): return self.data.train_dataloader()
    @property
    def valid_dl(self): return self.data.val_dataloader()
    @property
    def test_dl(self): return self.data.test_dataloader()
    @property
    def predcit_dl(self): return self.data.predict_dataloader()
    @property
    def y_secid(self) -> np.ndarray: return self.data.y_secid
    @property
    def y_date(self) -> np.ndarray: return self.data.y_date
    @property
    def model_dict(self): return self.model.to_dict()
    @property
    def is_cuda(self) -> bool: return self.module.device.device.type == 'cuda'
    @property
    def booster_train_params(self): return {'seed' : self.module.config['random_seed'] , **self.module.model_param}
    @property
    def batch_data(self) -> BatchData:
        assert isinstance(self.module.batch_data , BatchData)
        return self.module.batch_data
    
    def reset(self): self.loaded = False
    def load(self , model_dict : dict):
        '''load self.model'''
        self.model = GeneralBooster.from_dict(model_dict)
        self.loaded = True
        return self
    
    def booster_input(self , dataset : Literal['train' , 'valid' , 'test'] = 'test' , *args , **kwargs):
        if dataset == 'train':   booster_input = self.train_booster_input(*args , **kwargs)
        elif dataset == 'valid': booster_input = self.valid_booster_input(*args , **kwargs)
        elif dataset == 'test':  booster_input = self.test_booster_input(*args , **kwargs)
        return booster_input

    def train_booster_input(self , *args , **kwargs) -> BoosterInput | Any:
        assert self.module.status.dataset in ['train','valid'] , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , tuple) and len(batch_data) == 2 and isinstance(batch_data[0] , BoosterInput) , batch_data
        return batch_data[0]
    
    def valid_booster_input(self , *args , **kwargs) -> BoosterInput | Any:
        assert self.module.status.dataset in ['train','valid'] , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , tuple) and len(batch_data) == 2 and isinstance(batch_data[1] , BoosterInput) , batch_data
        return batch_data[1]
    
    def test_booster_input(self , *args , **kwargs) -> BoosterInput | Any:
        assert self.module.status.dataset == 'test' , self.module.status.dataset
        batch_data = self.module.batch_data
        assert isinstance(batch_data , BoosterInput)
        return batch_data

    def fit(self , *args , **kwargs):
        train_data = self.booster_input('train', *args , **kwargs)
        valid_data = self.booster_input('valid', *args , **kwargs)
        self.model = GeneralBooster(get_booster_type(self.module.config) , train_param=self.booster_train_params , 
                                    train = train_data , valid = valid_data , cuda=self.is_cuda).fit()
        self.loaded = True
        return self
    
    def predict(self , dataset : Literal['train' , 'valid' , 'test'] = 'test' , *args , **kwargs):
        booster_input = self.booster_input(dataset , *args , **kwargs)
        pred  = self.model.predict(booster_input).to_2d()
        output = BatchOutput((pred , {'label' : booster_input.y}))
        return output
