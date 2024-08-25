
from abc import ABC , abstractmethod
from torch import nn , Tensor
from typing import Any , Optional

from .booster import BoosterModel
from .ensembler import choose_net_ensembler , BoosterEnsembler
from ...nn import GetNN
from ...util import Checkpoint , Metrics , TrainConfig , BaseDataModule , BaseTrainer , ModelDict , ModelFile

class ModelEnsembler(ABC):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainer , use_score = True , **kwargs) -> None:
        self.module = model_module
        self.use_score = use_score

    @property
    def config(self) -> TrainConfig: return self.module.config
    @property
    def data(self) -> BaseDataModule: return self.module.data
    @property
    def ckpt(self) -> Checkpoint: return self.module.checkpoint
    @property
    def device(self): return self.module.device

    @classmethod
    def setup(cls , model_module : BaseTrainer , **kwargs): 
        '''if model_module is booster , use booster'''
        assert isinstance(model_module.config , TrainConfig)
        if model_module.config.module_type == 'booster':
            return BoosterManager(model_module , **kwargs)
        elif model_module.config.module_type == 'aggregator':
            return AggregatorManager(model_module , **kwargs)
        else:
            return NetManager(model_module , **kwargs)
        
    @classmethod
    def get_net(cls , model_module : str , model_param = {} , state_dict : Optional[dict[str,Tensor]] = {} , device : Any = None):
        NN = GetNN(model_module)
        net = NN.nn_module(**model_param)
        assert isinstance(net , nn.Module) , net.__class__
        if state_dict: net.load_state_dict(state_dict)
        return device(net) if callable(device) else net.to(device)
    
    @classmethod
    def get_model(cls , config : TrainConfig , model : ModelDict | ModelFile , 
                  model_param : dict | None = None , model_num : int | None = None , device : Any = None):
        if config.module_type == 'nn':
            if model_param is None:
                assert model_num is not None
                model_param = config.model_param[model_num]
            base_net = GetNN(config.model_module).nn_module(**model_param)
        else:
            base_net = None
        model_instance = model.model_instance(base_net , device)
        return model_instance
    
    @abstractmethod
    def new_model(self , training : bool , model_file : ModelFile): 
        '''call when fitting/testing new model'''
        return self
    @abstractmethod
    def model(self , *args , **kwargs) -> Any: '''model object that can be called to forward'''
    @abstractmethod
    def override(self): 
        '''
        override some component of self.module, called after test batch forward, before metrics
        e.g. booster head component of nn should be calculated and override net output in test stage
        '''
    @abstractmethod
    def assess(self , epoch : int , metrics : Metrics): '''update by assessing batch outout, called on_validation_epoch_end'''
    @abstractmethod
    def collect(self , model_type = 'best' , *args) -> ModelDict: '''update by assessing batch outout, called before stacking model'''

class NetManager(ModelEnsembler):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainer , use_score = True , **kwargs) -> None:
        super().__init__(model_module , use_score)
        self.net_ensemblers = {
            model_type:choose_net_ensembler(model_type)(self.ckpt , use_score , **kwargs)
            for model_type in self.config.model_types
        }
        if self.config.booster_head: 
            self.booster_head = BoosterEnsembler(model_module)
        else:
            self.booster_head = None

    def new_model(self , training : bool , model_file : ModelFile):
        for model in self.net_ensemblers.values(): model.reset()
        if not training and self.booster_head: self.booster_head.load(model_file['booster_head'])
        return self
    
    def model(self , use_state_dict = None , *args , **kwargs): return self.net(use_state_dict)
    def net(self , use_state_dict = None):
        return self.get_net(self.config.model_module, self.module.model_param, use_state_dict, self.device)

    def override(self):
        if self.booster_head: self.module.batch_output.override_pred(self.booster_head.predict().pred)

    def assess(self , epoch : int , metrics : Metrics): 
        for model in self.net_ensemblers.values():  model.assess(self.module.net , epoch , metrics)

    def collect(self , model_type = 'best' , *args):
        net = self.net_ensemblers[model_type].collect(self.module , *args)
        model_dict = ModelDict(state_dict = net.state_dict())
        if self.booster_head: model_dict.booster_head = self.booster_head.fit(net).model_dict
        return model_dict

class BoosterManager(ModelEnsembler):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainer , use_score = True , **kwargs) -> None:
        super().__init__(model_module , use_score)
        self.booster = BoosterModel(model_module)

    def new_model(self , training : bool , model_file : ModelFile):
        if not training: self.booster.load(model_file['booster_dict'])
        return self
    def model(self, *args , **kwargs) -> Any: return self.booster
    def override(self): ...
    def assess(self , epoch : int , metrics : Metrics): ...
    def collect(self , model_type = 'best' , *args):
        return ModelDict(booster_dict = self.booster.fit().model_dict)
    
class AggregatorManager(ModelEnsembler):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainer , use_score = True , **kwargs) -> None:
        super().__init__(model_module , use_score)
        self.booster = BoosterModel(model_module)

    def new_model(self , training : bool , model_file : ModelFile):
        if not training: self.booster.load(model_file['aggregator_dict'])
        return self
    def model(self, *args , **kwargs) -> Any: return self.booster
    def override(self): ...
    def assess(self , epoch : int , metrics : Metrics): ...
    def collect(self , model_type = 'best' , *args):
        return ModelDict(aggregator_dict = self.booster.fit().model_dict)