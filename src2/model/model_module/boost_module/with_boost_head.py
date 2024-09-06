from abc import ABC , abstractmethod

from torch import nn , Tensor
from typing import Any , Optional

from ...util import BatchData , BatchOutput , Checkpoint , ModelDict , ModelFile , TrainConfig
from ...util.classes import BasePredictorModel

class NNPredictor(BasePredictorModel):
    '''a group of ensemble models , of same net structure'''
    def __init__(self, trainer : BaseTrainer , **kwargs) -> None:
        self.trainer = trainer

    def __init__(self, model_module : BaseTrainer , use_score = True , **kwargs) -> None:
        super().__init__(model_module , use_score)
        self.net_ensemblers = {
            model_type:choose_swa_method(model_type)(self.ckpt , use_score , **kwargs)
            for model_type in self.config.model_submodels
        }
        if self.config.model_booster_head: 
            self.booster_head = BoosterEnsembler(model_module)
        else:
            self.booster_head = None

    @classmethod
    def setup(cls , model_module : BaseTrainer , **kwargs): 
        '''if model_module is booster , use booster'''
        assert isinstance(model_module.config , TrainConfig)
        if model_module.config.module_type == 'boost':
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

    @property
    def config(self) -> TrainConfig: return self.trainer.config
    @property
    def data(self) -> BaseDataModule: return self.trainer.data
    @property
    def ckpt(self) -> Checkpoint: return self.trainer.checkpoint
    @property
    def device(self): return self.trainer.device
    def __call__(self , input : BatchData | Tensor | Any):
        if input is None or len(input) == 0:
            output = None
        else:
            output = self.forward(input)
        return BatchOutput(output)
    
    @abstractmethod
    def load_model(self , training : bool , model_file : ModelFile): 
        '''call when fitting/testing new model'''
        return self
    @abstractmethod
    def forward(self , batch_data : BatchData | Tensor , *args , **kwargs) -> Any: 
        '''model object that can be called to forward'''
    @abstractmethod
    def fit(self):
        '''fit the model inside'''
        return self
    @abstractmethod
    def test(self) -> Any:
        '''test the model inside'''
    @abstractmethod
    def assess(self , epoch : int , *args , **kwargs) -> Any: 
        '''update by assessing batch outout, called on_validation_epoch_end'''
    @abstractmethod
    def collect(self , submodel = 'best' , *args) -> ModelDict: 
        '''update by assessing batch outout, called before stacking model'''
