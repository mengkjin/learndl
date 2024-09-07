import torch
import numpy as np

from torch import Tensor
from typing import Any , Iterator , Literal

from .basic import DataModule , TrainerModule
from ..classes import BoosterInput
from ..ensemble import BoosterModel
from ..util import DataloaderStored
from ...basic import PATH
from ...func import match_values
    
class BoosterDataModule(DataModule):
    '''for boosting such as algo.boost.lgbm, create booster'''
    def train_dataloader(self) -> Iterator[BoosterInput]: return self.loader_dict['train']
    def val_dataloader(self) -> Iterator[BoosterInput]:   return self.loader_dict['valid']
    def test_dataloader(self) -> Iterator[BoosterInput]:  return self.loader_dict['test']
    def predict_dataloader(self) -> Iterator[BoosterInput]: return self.loader_dict['test']
        
    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , 
              param = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1) -> None:
        super().setup(stage , param , model_date , True)

    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w = None , valid = None) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        index0, index1 = torch.arange(len(y)) , self.step_idx
        sample_index = self.split_sample(self.stage , index0 , index1 , self.config.train_train_ratio)
        self.storage.del_group(self.stage)
        assert len(x) == 1 , len(x)
        mdt0 , x0 = [(k,v) for k,v in x.items()][0]
        for set_key , set_samples in sample_index.items():
            if set_key in ['train' , 'valid']:
                bb_x , bb_y , bb_d = [] , [] , []
                for bnum , b_i in enumerate(set_samples):
                    assert torch.isin(b_i[:,1] , index1).all()
                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    b_x = self.prenorm(self.rolling_rotation(x0,self.seqs[mdt0],i0,i1) , mdt0) # [n_stock x seq_len x n_feat]
                    b_y = y[i0 , yindex1] # [n_stock x num_output]
                    assert b_x.shape[:2] == (len(self.y_secid) , self.seqs[mdt0]) , (b_x.shape , (len(self.y_secid) , self.seqs[mdt0]))
                    assert b_y.shape == (len(self.y_secid) , 1) , (b_y.shape , (len(self.y_secid) , 1))
                    assert all(yindex1 == yindex1[0]) , yindex1
                    b_x = b_x.reshape(len(self.y_secid),1,-1)

                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    bb_x.append(b_x)
                    bb_y.append(b_y)
                    bb_d.append(self.y_date[index1[yindex1[0]]])

                bb_x = torch.concat(bb_x , dim = 1)
                bb_y = torch.concat(bb_y , dim = 1)
                bb_d = np.array(bb_d)
                bnum = 0
                batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt']
                self.storage.save(BoosterInput.from_tensor(bb_x , bb_y , self.y_secid , bb_d) , batch_files[bnum] , group = self.stage)
            elif set_key == 'test':
                batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
                for bnum , b_i in enumerate(set_samples):
                    assert torch.isin(b_i[:,1] , index1).all()
                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    b_x = self.prenorm(self.rolling_rotation(x0,self.seqs[mdt0],i0,i1) , mdt0) # [n_stock x seq_len x n_feat]
                    b_y = y[i0 , yindex1] # [n_stock x num_output]
                    assert b_x.shape[:2] == (len(self.y_secid) , self.seqs[mdt0]) , (b_x.shape , (len(self.y_secid) , self.seqs[mdt0]))
                    assert b_y.shape == (len(self.y_secid) , 1) , (b_y.shape , (len(self.y_secid) , 1))
                    assert all(yindex1 == yindex1[0]) , yindex1
                    b_x = b_x.reshape(len(self.y_secid),1,-1)
                    dates = np.array([self.y_date[index1[yindex1[0]]]])
                    self.storage.save(BoosterInput.from_tensor(b_x , b_y , self.y_secid , dates) , batch_files[bnum] , group = self.stage)
            else:
                raise KeyError(set_key)
            self.loader_dict[set_key] = DataloaderStored(self.storage , batch_files)

    @staticmethod
    def split_sample(stage , index0 : Tensor , index1 : Tensor , train_ratio   : float = 0.8) -> dict[str,list]:
        l0 , l1 = len(index0) , len(index1)
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)

        def sequential_sampling(beg , end , posit = pos): return [posit[:,j] for j in range(beg , end)]
        
        sample_index = {}
        if stage == 'fit':
            # must be sequential
            sep = int(l1 * train_ratio)
            sample_index['train'] = sequential_sampling(0 , sep)
            sample_index['valid'] = sequential_sampling(sep , l1)
        else:
            # test dataloader should have the same length as dates, so no filtering of val[:,j].sum() > 0
            sample_index['test'] = sequential_sampling(0 , l1)
        return sample_index

class BoosterTrainer(TrainerModule):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data : BoosterDataModule = BoosterDataModule(self.config)

    def batch_forward(self) -> None: 
        if self.status.dataset == 'train': self.booster.fit(silence=True)
        self.batch_output = self.booster.predict(self.status.dataset)

    def batch_metrics(self) -> None:
        self.metrics.calculate_from_tensor(self.status.dataset , self.batch_output.other['label'] , self.batch_output.pred, assert_nan = True)
        self.metrics.collect_batch_metric()

    def batch_backward(self) -> None: ...

    def fit_model(self):
        self.on_fit_model_start()
        for self.batch_idx , self.batch_data in enumerate(zip(self.data.train_dataloader() , self.data.val_dataloader())):
            self.status.dataset_train()
            self.on_train_batch_start()
            self.on_train_batch()
            self.on_train_batch_end()

            self.status.dataset_validation()
            self.on_validation_batch_start()
            self.on_validation_batch()
            self.on_validation_batch_end()

        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        self.status.dataset_test()
        for self.status.model_type in self.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.data.test_dataloader()):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    def on_configure_model(self):  self.config.set_config_environment()
    def on_fit_model_start(self):
        self.data.setup('fit' , self.model_param , self.model_date)
        self.load_model(True)
    def on_fit_model_end(self): self.save_model()
    
    def on_train_batch_start(self):
        self.metrics.new_epoch_metric('train' , self.status)

    def on_train_batch_end(self): 
        self.metrics.collect_epoch_metric('train')

    def on_validation_batch_start(self):
        self.metrics.new_epoch_metric('valid' , self.status)

    def on_validation_batch_end(self): 
        self.metrics.collect_epoch_metric('valid')

    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.metrics.new_epoch_metric('test' , self.status)

    def on_test_model_type_end(self): 
        self.metrics.collect_epoch_metric('test')

    def on_test_model_start(self):
        if not self.deposition.exists(self.model_date , self.model_num , self.model_type): self.fit_model()
        self.data.setup('test' , self.model_param , self.model_date)

    def on_test_batch(self):
        if self.batch_idx < self.batch_warm_up: return
        self.batch_forward()
        self.batch_metrics()
    
    def load_model(self , training : bool , *args , **kwargs):
        '''load model state dict, return net and a sign of whether it is transferred'''
        model_file = self.deposition.load_model(self.model_date , self.model_num)
        self.booster : BoosterModel = self.model.new_model(training , model_file).model()
        self.metrics.new_model(self.model_param)

    def stack_model(self):
        self.on_before_save_model()
        for model_type in self.model_types:
            model_dict = self.model.collect(model_type)
            self.deposition.stack_model(model_dict , self.model_date , self.model_num , model_type) 

    def save_model(self):
        self.stack_model()
        for model_type in self.model_types:
            self.deposition.dump_model(self.model_date , self.model_num , model_type) 
    
    def __call__(self , input : BoosterInput): raise Exception('Undefined call')

