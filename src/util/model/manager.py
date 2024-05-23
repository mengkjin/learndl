
from abc import ABC , abstractmethod
from torch import nn , Tensor
from typing import Any , Literal , Optional

from .net import choose_ensembler
from .boost import LgbmEnsembler , LgbmBooster
from ..config import TrainConfig
from ..metric import Metrics
from ..store import Checkpoint
from ... import nn as NN
from ...classes import BaseDataModule , BaseTrainerModule , ModelDict , ModelFile
from ...environ import BOOSTER_MODULE


class ModelManager(ABC):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainerModule , use_score = True , **kwargs) -> None:
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
    def setup(cls , model_module : BaseTrainerModule , **kwargs): 
        '''if model_module is in BOOSTER_MODULE(e.g. "lgbm"), use booster'''
        assert isinstance(model_module.config , TrainConfig)
        if model_module.config.model_module in BOOSTER_MODULE:
            return BoosterManager(model_module , **kwargs)
        else:
            return NetManager(model_module , **kwargs)
        
    @classmethod
    def get_net(cls , model_module : str , model_param = {} , state_dict : Optional[dict[str,Tensor]] = {} , device : Any = None):
        net = getattr(NN , model_module)(**model_param)
        assert isinstance(net , nn.Module) , net.__class__
        if state_dict: net.load_state_dict(state_dict)
        return device(net) if callable(device) else net.to(device)

    @abstractmethod
    def new_model(self , training : bool , model_file : ModelFile): '''call when fitting/testing new model'''
    @abstractmethod
    def model(self , *args , **kwargs): '''model object that can be called to forward'''
    @abstractmethod
    def override(self): '''override some componet of self.module, called after test batch forward, before metrics'''
    @abstractmethod
    def assess(self , epoch : int , metrics : Metrics): '''update by assessing batch outout, called on_validation_epoch_end'''
    @abstractmethod
    def collect(self , model_type = 'best' , *args): '''update by assessing batch outout, called before stacking model'''


class NetManager(ModelManager):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainerModule , use_score = True , **kwargs) -> None:
        super().__init__(model_module , use_score)
        self.net_ensemblers = {
            model_type:choose_ensembler(model_type)(self.ckpt , use_score , **kwargs)
            for model_type in self.config.model_types
        }
        if self.config.lgbm_ensembler: 
            self.net_lgbm_head = LgbmEnsembler(model_module)
        else:
            self.net_lgbm_head = None

    def new_model(self , training : bool , model_file : ModelFile):
        for model in self.net_ensemblers.values(): model.reset()
        if not training and self.net_lgbm_head: self.net_lgbm_head.load(model_file['lgbm_string'])
        return self
    
    def model(self , use_state_dict = None , *args , **kwargs): return self.net(use_state_dict)
    def net(self , use_state_dict = None):
        return self.get_net(self.config.model_module, self.module.model_param, use_state_dict, self.device)

    def override(self):
        if self.net_lgbm_head: self.module.batch_output.override_pred(self.net_lgbm_head.predict())

    def assess(self , epoch : int , metrics : Metrics): 
        for model in self.net_ensemblers.values():  model.assess(self.module.net , epoch , metrics)

    def collect(self , model_type = 'best' , *args):
        model_dict = ModelDict()
        net = self.net_ensemblers[model_type].collect(self.module.net , self.data , *args , device=self.device)
        model_dict.state_dict = net.state_dict()
        if self.net_lgbm_head: model_dict.lgbm_string = self.net_lgbm_head.fit(net).model_string
        return model_dict

class BoosterManager(ModelManager):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseTrainerModule , use_score = True , **kwargs) -> None:
        super().__init__(model_module , use_score)
        self.booster_lgbm = LgbmBooster(model_module)

    def new_model(self , training : bool , model_file : ModelFile):
        if not training: self.booster_lgbm.load(model_file['lgbm_string'])
        return self
    def model(self, *args , **kwargs) -> Any: return self.booster()
    def booster(self): return self.booster_lgbm
    def override(self): ...
    def assess(self , epoch : int , metrics : Metrics): ...
    def collect(self , model_type = 'best' , *args):
        model_dict = ModelDict()
        model_dict.lgbm_string = self.booster_lgbm.fit().model_string
        return model_dict
