import gc , torch
import numpy as np
import pandas as pd

from torch import Tensor
from typing import Any , Iterator , Literal

from .basic import DataModule , TrainerModule
from ..models import BoosterModel
from ...util import DataloaderStored , BoosterInput , BatchOutput
from ....basic import PATH , CONF
from ....data import ModuleData , DataBlock
from ....func import index_intersect , match_values
    

class AggregatorDataModule(DataModule):
    '''for boosting such as algo.boost.lgbm, create booster'''
    def train_dataloader(self) -> Iterator[BoosterInput]: return self.loader_dict['train']
    def val_dataloader(self) -> Iterator[BoosterInput]:   return self.loader_dict['valid']
    def test_dataloader(self) -> Iterator[BoosterInput]:  return self.loader_dict['test']
    def predict_dataloader(self) -> Iterator[BoosterInput]: return self.loader_dict['test']
        
    def load_data(self):
        #with CONF.Silence():
        self.datas = ModuleData.load([] , self.config.model_data_labels, 
                                     fit = self.use_data != 'predict' , predict = self.use_data != 'fit' ,
                                     dtype = self.config.precision)
        self.labels_n = min(self.datas.y.shape[-1] , self.config.Model.max_num_output)
        if self.use_data == 'predict':
            self.model_date_list = self.datas.date[0]
            self.test_full_dates = self.datas.date[1:]
        else:
            self.model_date_list = self.datas.date_within(self.config['beg_date'] , self.config['end_date'] , self.config['interval'])
            self.test_full_dates = self.datas.date_within(self.config['beg_date'] , self.config['end_date'])[1:]

        self.static_prenorm_method = {}
        self.reset_dataloaders()
        return self

    def setup(self, stage : Literal['fit' , 'test' , 'predict'] , 
              param = {'seqlens' : {'day': 30 , '30m': 30 , 'style': 30}} , 
              model_date = -1) -> None:
        if self.use_data == 'predict': stage = 'predict'

        if self.loader_param == (stage , model_date): return
        self.loader_param = stage , model_date

        assert stage in ['fit' , 'test' , 'predict'] and model_date > 0 , (stage , model_date)

        self.stage = stage
        self.seqs = {'hidden':1}
        self.seq0 = self.seqx = self.seqy = 1

        hidden_dates : list[np.ndarray] = []
        hidden_df : pd.DataFrame | Any = None
        ds_list = ['train' , 'valid'] if stage == 'fit' else ['test' , 'predict']
        for hidden_key in self.config.model_data_hiddens:
            model_name , model_num , model_type = hidden_key.split('.')
            hidden_path = PATH.hidden.joinpath(model_name , f'hidden.{model_num}.{model_type}.{model_date}.feather')
            df = pd.read_feather(hidden_path)
            df = df[df['dataset'].isin(ds_list)].drop(columns='dataset').set_index(['secid','date'])
            hidden_dates.append(df.index.get_level_values('date').unique().to_numpy())
            df.columns = [f'{hidden_key}.{col}' for col in df.columns]
            hidden_df = df if hidden_df is None else hidden_df.join(df , how='outer')

        stage_date = index_intersect(hidden_dates)[0]
        if self.stage != 'fit':
            stage_date = index_intersect([stage_date , self.test_full_dates])[0]
        self.day_len = len(stage_date)
        self.step_len = len(stage_date)
        self.date_idx , self.step_idx = torch.arange(self.day_len) , torch.arange(self.day_len)

        y_aligned = self.datas.y.align_date(stage_date , inplace=False)
        self.y_secid , self.y_date = y_aligned.secid , y_aligned.date

        if stage == 'fit':
            ...
        elif stage in ['predict' , 'test']:
            self.model_test_dates = stage_date
            self.early_test_dates = stage_date[:0]
        else:
            raise KeyError(stage)

        x = {'hidden':DataBlock.from_dataframe(hidden_df).align_secid_date(self.y_secid , self.y_date).as_tensor().values}
        y = Tensor(y_aligned.values).squeeze(2)[...,:self.labels_n]
        self.hidden_cols = hidden_df.columns
        self.y , _ = self.standardize_y(y , None , None , no_weight = True)

        if stage != 'fit':
            w , valid = None , None
            y , _ = self.standardize_y(self.y , None , self.step_idx)
        else:
            valid = self.full_valid_sample(x , self.y , self.step_idx)
            y , w = self.standardize_y(self.y , valid , self.step_idx)

        self.y[:,self.step_idx] = y[:]
        self.static_dataloader(x , y , w , valid)

        gc.collect() 
        torch.cuda.empty_cache()

    def static_dataloader(self , x : dict[str,Tensor] , y : Tensor , w = None , valid = None) -> None:
        '''update loader_dict , save batch_data to f'{PATH.model}/{model_name}/{set_name}_batch_data' and later load them'''
        index0, index1 = torch.arange(len(y)) , self.step_idx
        sample_index = self.split_sample(self.stage , index0 , index1 , self.config.train_ratio)
        self.storage.del_group(self.stage)
        assert len(x) == 1 , len(x)
        x0 = x['hidden']
        for set_key , set_samples in sample_index.items():
            if set_key in ['train' , 'valid']:
                bb_x , bb_y , bb_d = [] , [] , []
                for bnum , b_i in enumerate(set_samples):
                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    bb_x.append(x0[i0 , i1].reshape(len(self.y_secid),1,-1))
                    bb_y.append(y[i0 , yindex1])
                    bb_d.append(self.y_date[index1[yindex1[0]]])
                bb_x = torch.concat(bb_x , dim = 1)
                bb_y = torch.concat(bb_y , dim = 1)
                bb_d = np.array(bb_d)
                bnum = 0
                batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt']
                self.storage.save(BoosterInput.from_tensor(bb_x , bb_y , self.y_secid , bb_d , self.hidden_cols) , batch_files[bnum] , group = self.stage)
            elif set_key == 'test':
                batch_files = [f'{PATH.batch}/{set_key}.{bnum}.pt' for bnum in range(len(set_samples))]
                for bnum , b_i in enumerate(set_samples):
                    i0 , i1 , yindex1 = b_i[:,0] , b_i[:,1] , match_values(b_i[:,1] , index1)

                    b_x = x0[i0,i1].reshape(len(self.y_secid),1,-1)
                    b_y = y[i0 , yindex1] # [n_stock x num_output]
                    dates = np.array([self.y_date[index1[yindex1[0]]]])
                    self.storage.save(BoosterInput.from_tensor(b_x , b_y , self.y_secid , dates , self.hidden_cols) , batch_files[bnum] , group = self.stage)
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
    
class AggregatorTrainer(TrainerModule):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data : AggregatorDataModule = AggregatorDataModule(self.config)

    def batch_forward(self) -> None: 
        if self.status.dataset == 'train': self.booster.fit(silence = True)
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

    