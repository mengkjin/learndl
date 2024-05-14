import numpy as np
import torch

from abc import ABC , abstractmethod
from copy import deepcopy
from torch import nn
from torch.optim.swa_utils import AveragedModel , update_bn
from typing import Iterator , Literal , Optional

from .store import Checkpoint
from .config import TrainConfig
from ..classes import BatchData , BatchOutput , BoosterData , BaseDataModule , NdData
from ..algo.boost.lgbt import Lgbt

class EnsembleModels:
    '''a group of ensemble models , of same net structure'''
    def __init__(self, net : nn.Module , config : TrainConfig , data_mod : BaseDataModule , 
                 ckpt : Checkpoint , use_score = True , device = None , **kwargs) -> None:
        self.net    = net
        self.config = config
        self.data_mod = data_mod
        self.models = {model_type:self.choose(model_type)(ckpt , use_score , **kwargs) for model_type in config.model_types}
        self.device = device
        #if self.config.lgbt_ensembler:
        #    self.lgbt_ensembler = LgbtEnsembler(self.net , self.data_mod , self.device)
        
    def __getitem__(self , key): 
        return self.models[key]
    def assess(self , net , epoch : int , score = 0. , loss = 0.): 
        for model in self.models.values(): model.assess(net , epoch , score , loss)
    def collect(self , model_type , *args):
        model_dict = {}
        #net = self.models[model_type].collect(self.net , self.data_mod , *args , device = self.device)
        #model_dict['state_dict'] = net.state_dict()

        model_dict['state_dict'] = self.models[model_type].collect(self.net , self.data_mod , *args , device = self.device)

        #if self.config.lgbt_ensembler:
        #    model_dict['lgbt_model_string'] = self.lgbt_ensembler.fit().model_string
            # print(model_dict['lgbt_model_string'])
        return model_dict
    
    @staticmethod
    def choose(model_type):
        '''get a subclass of _BaseEnsembleModel'''
        if model_type == 'best': return BestModel
        elif model_type == 'swabest': return SWABest
        elif model_type == 'swalast': return SWALast
        else: raise KeyError(model_type)

class _BaseEnsembleModel(ABC):
    '''abstract class of fittest model, e.g. model with the best score, swa model of best scores or last ones'''
    def __init__(self, ckpt : Checkpoint , use_score = True , **kwargs) -> None:
        self.ckpt , self.use_score = ckpt , use_score
    @abstractmethod
    def assess(self , net , epoch : int , score = 0. , loss = 0.): '''score or loss to update assessment'''
    @abstractmethod
    def collect(self , net , *args , device = None , **kwargs): '''output the final fittest model state dict'''

class SWAModel:
    def __init__(self , module : nn.Module) -> None:
        self.template = deepcopy(module)
        self.avgmodel = AveragedModel(self.template)

    def update_sd(self , state_dict):
        self.template.load_state_dict(state_dict)
        self.avgmodel.update_parameters(self.template) 
        return self
    
    def update_bn(self , data_loader , device = None):
        self.avgmodel = device(self.avgmodel) if callable(device) else self.avgmodel.to(device)
        update_bn(self.bn_loader(data_loader) , self.avgmodel) 
        return self
     
    def bn_loader(self , data_loader):
        for batch_data in data_loader: 
            yield (batch_data.x , batch_data.y , batch_data.w)

    @property
    def module(self) -> nn.Module: return self.avgmodel.module

class BestModel(_BaseEnsembleModel):
    def __init__(self, ckpt : Checkpoint , use_score = True , **kwargs) -> None:
        super().__init__(ckpt , use_score)
        self.epoch_fix  = -1
        self.metric_fix = None

    def assess(self , net , epoch : int , score = 0. , loss = 0.):
        value = score if self.use_score else loss
        if self.metric_fix is None or (self.metric_fix < value if self.use_score else self.metric_fix > value):
            self.ckpt.disjoin(self , self.epoch_fix)
            self.epoch_fix = epoch
            self.metric_fix = value
            self.ckpt.join(self , epoch , net)

    def collect(self , net : nn.Module , *args , **kwargs):
        return self.ckpt.load_epoch(self.epoch_fix)
        net = deepcopy(net)
        net.load_state_dict(self.ckpt.load_epoch(self.epoch_fix))
        return net

class SWABest(_BaseEnsembleModel):
    def __init__(self, ckpt : Checkpoint , use_score = True , n_best = 5 , **kwargs) -> None:
        super().__init__(ckpt , use_score)
        assert n_best > 0, n_best
        self.n_best      = n_best
        self.metric_list = []
        self.candidates  = []
        
    def assess(self , net , epoch : int , score = 0. , loss = 0.):
        value = score if self.use_score else loss
        if len(self.metric_list) == self.n_best :
            arg = np.argmin(self.metric_list) if self.use_score else np.argmax(self.metric_list)
            if (self.metric_list[arg] < value if self.use_score else self.metric_list[arg] > value):
                self.metric_list.pop(arg)
                candid = self.candidates.pop(arg)
                self.ckpt.disjoin(self , candid)

        if len(self.metric_list) < self.n_best:
            self.metric_list.append(value)
            self.candidates.append(epoch)
            self.ckpt.join(self , epoch , net)

    def collect(self , net , data_mod : BaseDataModule , *args , device = None , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: swa.update_sd(self.ckpt.load_epoch(epoch))
        loader = data_mod.train_dataloader()
        swa.update_bn(loader , getattr(loader , 'device' , device))
        return swa.module.cpu().state_dict()
    
class SWALast(_BaseEnsembleModel):
    def __init__(self, ckpt : Checkpoint , use_score = True , n_last = 5 , interval = 3 , **kwargs) -> None:
        super().__init__(ckpt , use_score)
        assert n_last > 0 and interval > 0, (n_last , interval)
        self.n_last      = n_last
        self.interval    = interval
        self.left_epochs = (n_last // 2) * interval
        self.epoch_fix   = -1
        self.metric_fix  = None
        self.candidates  = []

    def assess(self , net , epoch : int , score = 0. , loss = 0.):
        value = score if self.use_score else loss
        if self.metric_fix is None or (self.metric_fix < value if self.use_score else self.metric_fix > value):
            self.epoch_fix , self.metric_fix = epoch , value
        candidates = self._new_candidates(epoch)
        [self.ckpt.disjoin(self , candid) for candid in self.candidates if candid < min(candidates)]
        if epoch in candidates: self.ckpt.join(self , epoch , net)
        self.candidates = candidates

    def _new_candidates(self , epoch):
        epochs  = np.arange(self.interval , epoch + 1 , self.interval)
        left    = epochs[epochs < self.epoch_fix]
        right   = epochs[epochs > self.epoch_fix]
        return [*left[-((self.n_last - 1) // 2):] , self.epoch_fix , *right][:self.n_last]

    def collect(self , net , data_mod : BaseDataModule , *args , device = None , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: 
            swa.update_sd(self.ckpt.load_epoch(epoch))
        loader = data_mod.train_dataloader()
        swa.update_bn(loader , getattr(loader , 'device' , device))
        return swa.module.cpu().state_dict()
    
class LgbtEnsembler:
    '''load booster data and fit'''
    def __init__(self , net : nn.Module , data_mod : BaseDataModule , device = None) -> None:
        self.net = net
        self.data_mod = data_mod
        self.device = device

    @property
    def train_dl(self) -> Iterator[BatchData]: return self.data_mod.train_dataloader()
    @property
    def valid_dl(self) -> Iterator[BatchData]: return self.data_mod.val_dataloader()
    @property
    def test_dl(self) -> Iterator[BatchData]: return self.data_mod.test_dataloader()
    @property
    def y_secid(self) -> np.ndarray | torch.Tensor: return self.data_mod.y_secid
    @property
    def y_date(self) -> np.ndarray | torch.Tensor: return self.data_mod.y_date
    @property
    def model_string(self): return self.model.model_to_string()

    def booster_data(self , loader : Iterator[BatchData]) -> Optional[BoosterData]:
        hh , yy , ii = [] , [] , []
        device = getattr(loader , 'device' , self.device)
        self.net = device(self.net) if callable(device) else self.net.to(device)
        for batch_data in loader:
            hidden = BatchOutput(self.net(batch_data.x)).hidden
            if hidden is None: return 
            hh.append(hidden.detach().cpu().numpy())
            yy.append(batch_data.y.detach().cpu().numpy())
            ii.append(batch_data.i)
    
        hh , yy , ii = np.vstack(hh) , np.vstack(yy) , np.vstack(ii)
        secid_i , secid_j = np.unique(ii[:,0] , return_inverse=True)
        date_i  , date_j  = np.unique(ii[:,1] , return_inverse=True)
        hh_values = np.full((len(secid_i) , len(date_i) , hh.shape[-1]) , fill_value=np.nan)
        hh_values[secid_j , date_j] = hh[:]
        
        yy_values = np.full((len(secid_i) , len(date_i) , 1) , fill_value=np.nan)
        yy_values[secid_j , date_j] = yy[...,:1]

        return BoosterData(hh_values , yy_values , self.y_secid[secid_i] , self.y_date[date_i])

    def fit(self):
        self.net.eval()
        with torch.no_grad():
            train_data = self.booster_data(self.train_dl)
            valid_data = self.booster_data(self.valid_dl)
        self.model = Lgbt(train_data , valid_data).fit()
        # self.model.plot.training()
        return self
    
    def load(self , model_str):
        self.model = Lgbt.model_from_string(model_str)
        return self
    
    def predict(self , batch_data : BatchData , batch_output : BatchOutput):
        hidden = batch_output.hidden
        if hidden is None: return
        hidden = hidden.detach().cpu().numpy()
        label = batch_data.y.detach().cpu().numpy()
        
        new_pred = self.model.predict(BoosterData(hidden , label) , reform = False)
        print(new_pred)
        return new_pred
            
    