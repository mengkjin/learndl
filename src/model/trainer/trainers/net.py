import torch
import numpy as np

from numpy.random import permutation
from torch import Tensor
from torch.utils.data import BatchSampler
from typing import Any , Iterator , Literal , Optional

from .basic import DataModule , TrainerModule
from ...util import BatchData , DataloaderStored , LoaderWrapper , Optimizer
from ....basic import PATH
from ....func import match_values

class NetDataModule(DataModule):
    def train_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['train'] , self.device , self.config.verbosity)
    def val_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['valid'] , self.device , self.config.verbosity)
    def test_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['test'] , self.device , self.config.verbosity)
    def predict_dataloader(self):
        return LoaderWrapper(self , self.loader_dict['test'] , self.device , self.config.verbosity)
    def transfer_batch_to_device(self , batch : BatchData , device = None , dataloader_idx = None):
        return batch.to(self.device if device is None else device)

    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , 
              param = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1) -> None:
        super().setup(stage , param , model_date , False)
        
    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w : Optional[Tensor] , valid : Tensor) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        index0, index1 = torch.arange(len(valid)) , self.step_idx
        sample_index = self.split_sample(self.stage , valid , index0 , index1 , self.config.sample_method , 
                                         self.config.train_ratio , self.config['batch_size'])
        self.storage.del_group(self.stage)
        for set_key , set_samples in sample_index.items():
            assert set_key in ['train' , 'valid' , 'test'] , set_key
            shuf_opt = self.config.shuffle_option if set_key == 'train' else 'static'
            batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
            for bnum , b_i in enumerate(set_samples):
                assert torch.isin(b_i[:,1] , index1).all()
                i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)


                b_x = [self.prenorm(self.rolling_rotation(x[mdt],self.seqs[mdt],i0,i1) , mdt) for mdt in x.keys()]
                b_y = y[i0 , yindex1]
                b_w = None if w is None else w[i0 , yindex1]
                b_v = valid[i0 , yindex1]

                self.storage.save(BatchData(b_x , b_y , b_w , b_i , b_v) , batch_files[bnum] , group = self.stage)
            self.loader_dict[set_key] = DataloaderStored(self.storage , batch_files , shuf_opt)

    def split_sample(self , stage , valid : Tensor , index0 : Tensor , index1 : Tensor ,
                     sample_method : Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle'] = 'sequential' ,
                     train_ratio   : float = 0.8 , batch_size : int = 2000) -> dict[str,list]:
        '''
        update index of train/valid sub-samples of flattened all-samples(with in 0:len(index[0]) * step_len - 1)
        sample_tensor should be boolean tensor , True indicates non

        train/valid sample method: total_shuffle , sequential , both_shuffle , train_shuffle
        test sample method: sequential
        '''
        l0 , l1 = valid.shape[:2]
        pos = torch.stack([index0.repeat_interleave(l1) , index1.repeat(l0)] , -1).reshape(l0,l1,2)
        
        def shuffle_sampling(i , bs = batch_size):
            return [i[p] for p in BatchSampler(permutation(np.arange(len(i))) , bs , drop_last=False)]
        def sequential_sampling(beg , end , posit = pos , valid = valid):
            return [posit[:,j][valid[:,j]] for j in range(beg , end)]
        
        sample_index = {}
        if stage == 'fit':
            sep = int(l1 * train_ratio)
            if sample_method == 'total_shuffle':
                pool = torch.tensor(permutation(np.arange(valid.sum().item())))
                sep = int(len(pool) * train_ratio)
                sample_index['train'] = shuffle_sampling(pos[valid][pool[:sep]])
                sample_index['valid'] = shuffle_sampling(pos[valid][pool[sep:]])
            elif sample_method == 'both_shuffle':
                sample_index['train'] = shuffle_sampling(pos[:,:sep][valid[:,:sep]])
                sample_index['valid'] = shuffle_sampling(pos[:,sep:][valid[:,sep:]])
            elif sample_method == 'train_shuffle':
                sample_index['train'] = shuffle_sampling(pos[:,:sep][valid[:,:sep]])
                sample_index['valid'] = sequential_sampling(sep , l1)
            else:
                sample_index['train'] = sequential_sampling(0 , sep)
                sample_index['valid'] = sequential_sampling(sep , l1)
        else:
            # test dataloader should have the same length as dates, so no filtering of val[:,j].sum() > 0
            sample_index['test'] = sequential_sampling(0 , l1)
        return sample_index

class NetTrainer(TrainerModule):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data : NetDataModule = NetDataModule(self.config)
    def batch_forward(self) -> None: 
        self.batch_output = self(self.batch_data)
    def batch_metrics(self) -> None:
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
        self.metrics.calculate(self.status.dataset , self.batch_data, self.batch_output, self.net, assert_nan = True)
        self.metrics.collect_batch_metric()
    def batch_backward(self) -> None:
        if isinstance(self.batch_data , BatchData) and self.batch_data.is_empty: return
        assert self.status.dataset == 'train' , self.status.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.output)
        self.on_after_backward()

    def fit_model(self):
        self.status.fit_model_start()
        self.on_fit_model_start()
        while not self.status.end_of_loop:
            self.status.fit_epoch_start()
            self.on_fit_epoch_start()

            self.status.dataset_train()
            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()

            self.status.dataset_validation()
            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()

            self.on_before_fit_epoch_end()
            self.status.fit_epoch_end()
            self.on_fit_epoch_end()
        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        self.status.dataset_test()
        for self.status.model_type in self.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
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
    
    def on_train_epoch_start(self):
        self.net.train()
        torch.set_grad_enabled(True)
        self.dataloader = self.data.train_dataloader()
        self.metrics.new_epoch_metric('train' , self.status)
    
    def on_train_epoch_end(self): 
        self.metrics.collect_epoch_metric('train')
        self.optimizer.scheduler_step(self.status.epoch)
    
    def on_validation_epoch_start(self):
        self.net.eval()
        torch.set_grad_enabled(False)
        self.dataloader = self.data.val_dataloader()
        self.metrics.new_epoch_metric('valid' , self.status)
    
    def on_validation_epoch_end(self):
        self.metrics.collect_epoch_metric('valid')
        self.model.assess(self.status.epoch , self.metrics)
        torch.set_grad_enabled(True)
    
    def on_test_model_start(self):
        if not self.deposition.exists(self.model_date , self.model_num , self.model_type): self.fit_model()
        self.data.setup('test' , self.model_param , self.model_date)
        torch.set_grad_enabled(False)

    def on_test_model_end(self):
        torch.set_grad_enabled(True)
    
    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.dataloader = self.data.test_dataloader()
        self.assert_equity(len(self.dataloader) , len(self.batch_dates))
        self.metrics.new_epoch_metric('test' , self.status)
    
    def on_test_model_type_end(self): 
        self.metrics.collect_epoch_metric('test')

    def on_test_batch(self):
        self.assert_equity(self.batch_dates[self.batch_idx] , self.data.y_date[self.batch_data.i[0,1]]) 
        self.batch_forward()
        self.model.override()
        # before this is warmup stage , only forward
        if self.batch_idx < self.batch_warm_up: return
        self.batch_metrics()

    def on_before_save_model(self):
        self.net = self.net.cpu()
    
    def load_model(self , training : bool , model_type = 'best' , lr_multiplier = 1.):
        '''load model state dict, return net and a sign of whether it is transferred'''
        model_date = (self.prev_model_date if self.if_transfer else 0) if training else self.model_date
        model_file = self.deposition.load_model(model_date , self.model_num , model_type)
        self.transferred = training and self.if_transfer and model_file.exists()
        self.model = self.model.new_model(training , model_file)
        self.net : torch.nn.Module = self.model.model(model_file['state_dict'])
        self.metrics.new_model(self.model_param)
        if training:
            self.optimizer : Optimizer = Optimizer(self.net , self.config , self.transferred , lr_multiplier ,
                                                   model_module = self)
            self.checkpoint.new_model(self.model_param , self.model_date)
        else:
            assert model_file.exists() , str(model_file)
            self.net.eval()

    def stack_model(self):
        self.on_before_save_model()
        for model_type in self.model_types:
            model_dict = self.model.collect(model_type)
            self.deposition.stack_model(model_dict , self.model_date , self.model_num , model_type) 

    def save_model(self):
        if self.metrics.better_attempt(self.status.best_attempt_metric): self.stack_model()
        [self.deposition.dump_model(self.model_date , self.model_num , model_type) for model_type in self.model_types]