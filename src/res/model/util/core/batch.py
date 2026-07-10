"""
Batch input and output for the project
"""
from __future__ import annotations
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from dataclasses import dataclass , field
from functools import cached_property
from typing import Any

from src.proj import Proj , Logger , Base
from src.proj.util.functional.device import Device

__all__ = ['BatchInput' , 'BatchOutput' , 'BatchData']

def _object_shape(obj : Any) -> Any:
    if obj is None: 
        return None
    elif isinstance(obj , torch.Tensor | np.ndarray): 
        return obj.shape
    elif isinstance(obj , (list , tuple)): 
        return tuple([_object_shape(x) for x in obj])
    else: 
        return type(obj)
@dataclass
class BatchInput:
    """custom data component of a batch(x,y,w,i,effective)"""
    x       : torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    y       : torch.Tensor 
    w       : torch.Tensor | None
    i       : torch.Tensor 
    eff     : torch.Tensor
    y_date  : np.ndarray
    y_secid : np.ndarray
    kwargs  : dict[str,Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: 
            self.x = self.x[0]
        assert self.y is not None , 'y must not be None'
        assert self.i is not None , 'i must not be None'
        assert self.eff is not None , 'eff must not be None'
        assert self.w is None or self.w.shape == self.y.shape , (self.w.shape , self.y.shape)
        
    def to(self , device = None): 
        if device is None: 
            return self
        else:
            if isinstance(device , Device): 
                device = device.device
            inputs = {name:Device.send_to(getattr(self , name) , device) for name in ['x' , 'y' , 'w' , 'i' , 'eff' , 'y_date' , 'y_secid' , 'kwargs']}
            return BatchInput(**inputs)

    def check_x_integrity_for_nn(self , auto_fix = False):
        """
        check if x has nan, if yes, remove the rows with nan
        ! now default auto_fix as False, because ideally input of BatchInput should be valid_sampled first so have no nan
        ! careful, some prenormer if defined incorrectly might include new nans, so need to check and fix manually
        """
        if not auto_fix:
            return self
        if not self.x_has_nan:
            return self
        # self.auto_fix_nan_in_x()
        raise ValueError('Encountered nan in x for nn')

    def auto_fix_nan_in_x(self):
        if isinstance(self.x , torch.Tensor):
            nan_row = self.x.flatten(start_dim=1).isnan().any(dim=-1)
            self.x = self.x[~nan_row]
        else:
            nan_row = torch.stack([v.flatten(start_dim=1).isnan().any(dim=-1) for v in self.x], dim=-1).any(dim=-1)
            self.x = [v[~nan_row] for v in self.x]
        self.y = self.y[~nan_row]
        if self.w is not None:
            self.w = self.w[~nan_row]
        self.eff = self.eff[~nan_row]
        self.i = self.i[~nan_row]
        for key , value in self.kwargs.items():
            if not isinstance(value , torch.Tensor) or len(value) != len(nan_row):
                continue
            if value.ndim == 2 and value.shape[1] == len(value):
                self.kwargs[key] = value[nan_row][:,nan_row]
            else:
                self.kwargs[key] = value[nan_row]
        return self
        
    def cpu(self):  
        return self.to('cpu')
    def cuda(self): 
        return self.to('cuda')
    def mps(self): 
        return self.to('mps')
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, key) -> Any:
        if key == 'x': 
            return self.x
        elif key == 'y': 
            return self.y
        elif key == 'w': 
            return self.w
        elif key == 'i': 
            return self.i
        elif key == 'eff': 
            return self.eff
        else: 
            return self.kwargs[key]
    def keys(self):
        return ['x' , 'y' , 'w' , 'i' , 'eff' , *self.kwargs.keys()]
    def items(self):
        return {k:self[k] for k in self.keys()}
    
    @property
    def empty(self): 
        return len(self.y) == 0
    @property
    def device(self): 
        return self.y.device
    @property
    def shape(self): 
        return Base.shape(self , self.keys())
    @property
    def info(self):
        return f'{self.__class__.__name__}:\n' + \
            '\n'.join([f'{k} : {shape}' for k,shape in self.shape.items()])
    @property
    def x_has_nan(self):
        return self.x.isnan().any() if isinstance(self.x , torch.Tensor) else any(v.isnan().any() for v in self.x)

    @cached_property
    def date(self) -> np.ndarray:
        return self.y_date[self.i.cpu()[:,1].numpy()]
    
    @cached_property
    def secid(self) -> np.ndarray:
        return self.y_secid[self.i.cpu()[:,0].numpy()]
    
    @property
    def date0(self) -> int:
        assert np.all(self.date == self.date[0]) , self.date
        return self.date.astype(int)[0]

    @property
    def label0(self) -> np.ndarray:
        label = self.y.cpu().squeeze().numpy()
        if label.ndim == 1:
            return label
        elif label.ndim == 2:
            return label[:,0]
        else:
            raise ValueError(f'label shape {label.shape} is not supported')

    @classmethod
    def random(cls , batch_size = 2 , seq_len = 30 , n_inputs = 6 , predict_steps = 1):

        x = torch.rand(batch_size , seq_len , n_inputs)

        y = torch.rand(batch_size , predict_steps)
        w = None
        i = torch.Tensor([])
        v = torch.Tensor([])
        y_date = np.array([])
        y_secid = np.array([])
        return cls(x , y , w , i , v , y_date , y_secid)
    @classmethod
    def concat(cls , *batch_inputs):
        assert len(batch_inputs) > 0 , f'batch_inputs is empty'
        
        x , y , w , i , v , kwargs = [] , [] , [] , [] , [] , []
        y_date , y_secid = np.array([]), np.array([])
        for bd in batch_inputs:
            assert isinstance(bd , cls) , type(bd)
            x.append(bd.x)
            y.append(bd.y)
            w.append(bd.w)
            i.append(bd.i)
            v.append(bd.eff)
            if len(y_date) == 0:
                y_date = bd.y_date
            else:
                assert np.array_equal(y_date , bd.y_date) , (y_date , bd.y_date)
            if len(y_secid) == 0:
                y_secid = bd.y_secid
            else:
                assert np.array_equal(y_secid , bd.y_secid) , (y_secid , bd.y_secid)
            kwargs.append(bd.kwargs)
        assert all([len(kwg) == 0 for kwg in kwargs]) , [kwg.keys() for kwg in kwargs]
        if isinstance(x[0] , torch.Tensor):
            x = torch.concat(x)
        else:
            assert all([len(xx) == len(x[0]) for xx in x]) , [len(xx) for xx in x]
            x = type(x[0])([torch.concat([xx[j] for xx in x]) for j in range(len(x[0]))])
        y = torch.concat(y)
        assert all([type(ww) is type(w[0]) for ww in w]) , [type(ww) for ww in w]
        w = None if w[0] is None else torch.concat(w)
        i = torch.concat(i)
        v = torch.concat(v)
        return cls(x , y , w , i , v , y_date , y_secid)

    @classmethod
    def generate(cls , data_type : str = 'day+style' , label_num : int = 1):
        if data_type == 'random':
            return cls.random()
        else:
            from src.res.model.util import ModelConfig , DataModule
            assert label_num <= 2 , label_num
            override = {
                'model.module':'gru',
                'input.type':'data',
                'short_test':True,
                'input.data.types':data_type ,
                'model.labels': ['std_lag1_10' , 'std_lag1_20'] if label_num == 2 else ['std_lag1_10'],
                'num_output':[label_num],
                'input.sequence.steps':{'week':1},
                'input.sequence.lens':{'week':30},
            }
            with Proj.silence:
                model_config = ModelConfig(override=override)
                data = DataModule(model_config , 'predict').load_data()
                data.setup('predict' , model_date = data.datas.y.date[-20])
                batch_input = data.predict_dataloader()[0]
            Logger.stdout(batch_input.info)
            return batch_input
@dataclass
class BatchOutput:
    outputs : torch.Tensor | tuple | list | Any | None = None
    def __post_init__(self):
        if isinstance(self.outputs , BatchOutput):
            self.outputs = self.outputs.outputs
    def __len__(self): return 0 if self.outputs is None else len(self.pred)
    def __getitem__(self, key) -> Any:
        if key == 'pred': 
            return self.pred
        else: 
            return self.other[key]
    def keys(self):
        return ['pred' , *self.other.keys()]
    def items(self):
        return {k:self[k] for k in self.keys()}
    @property
    def shape(self):
        return Base.shape(self , self.keys())
    @property    
    def empty(self): 
        return len(self) == 0
    @property
    def device(self): 
        return self.pred.device
    @property
    def pred(self) -> torch.Tensor:
        if self.outputs is None: 
            return torch.Tensor(size=(0,1)).requires_grad_()
        output = self.outputs[0] if isinstance(self.outputs , (list , tuple)) else self.outputs
        if output.ndim == 1: 
            output = output.unsqueeze(1)
        assert output.ndim == 2 , output.ndim
        return output

    @property
    def other(self) -> dict[str,Any]:
        if isinstance(self.outputs , (list , tuple)):
            assert len(self.outputs) == 2 , self.outputs
            assert isinstance(self.outputs[1] , dict) , type(self.outputs[1])
            return self.outputs[1]
        else:
            return {}
    @property
    def info(self):
        return f'{self.__class__.__name__}:\n' + \
            '\n'.join([f'{k} : {shape}' for k,shape in self.shape.items()])
    
    @property
    def hidden(self) -> torch.Tensor: return self.other['hidden']
       
    def override_pred(self , pred : torch.Tensor | None):
        assert self.outputs is not None , f'{self} has outputs None'
        assert pred is not None , f'{pred} is None'
        raw_pred = self.pred
        assert len(pred) == len(raw_pred) , (pred.shape , raw_pred.shape)
        pred = pred.reshape(*raw_pred.shape).to(raw_pred)
        if isinstance(self.outputs , (list , tuple)):
            self.outputs = [pred , *self.outputs[1:]]
        else:
            self.outputs = pred
        return self
    
    def pred_df(
        self , secid : Base.alias.SecidType , date : Base.alias.DateType , * , 
        narrow_df = False , colnames : Base.alias.NamesType = None , colname_prefix : str = '' ,
        **kwargs
    ):
        pred = self.pred.cpu().numpy()
        if pred.ndim == 1: 
            pred = pred[:,None]
        assert pred.ndim == 2 , pred.shape

        columns = Base.ensure_name_list(colnames,[f'pred.{i}' for i in range(pred.shape[-1])])
        if colname_prefix:
            columns = [f'{colname_prefix}.{col}' for col in columns]
        assert pred.shape[-1] == len(columns) , (pred.shape , columns)

        df = pd.DataFrame(
            {'secid' : Base.ensure_secid(secid,[]) , 'date' : Base.ensure_date(date,[]) , 
            **{col:pred[:,i] for i,col in enumerate(columns)}}
        )
        if isinstance(colnames , str):
            assert pred.shape[-1] == 1 , (pred.shape , colnames)
            df = df.rename(columns={'pred.0':colnames})
        if narrow_df: 
            df = df.melt(['secid','date'] , var_name='feature')
        return df.assign(**kwargs)
        
    def hidden_df(
        self , secid : Base.alias.SecidType , date : Base.alias.DateType , * , 
        narrow_df = False , colnames : Base.alias.NamesType | None = None , colname_prefix : str = '' ,
        **kwargs
    ):
        """kwargs will be used in df.assign(**kwargs)"""
        full_hidden : torch.Tensor | Any = self.other['hidden']
        full_hidden = full_hidden.cpu().numpy()

        assert full_hidden.ndim == 2 , full_hidden.shape

        columns = Base.ensure_name_list(colnames,[f'hidden.{i}' for i in range(full_hidden.shape[-1])])
        if colname_prefix:
            columns = [f'{colname_prefix}.{col}' for col in columns]
        assert full_hidden.shape[-1] == len(columns) , (full_hidden.shape , columns)

        df = pd.DataFrame(
            {'secid' : Base.ensure_secid(secid,[]) , 'date' : Base.ensure_date(date,[]) , 
            **{col:full_hidden[:,i] for i,col in enumerate(columns)}}
        )
        if narrow_df: 
            df = df.melt(['secid','date'] , var_name='feature')
        return df.assign(**kwargs)
    
    @classmethod
    def from_module(cls , module : nn.Module | Any , inputs : BatchInput | BatchData | Any , **kwargs):
        """module can be a nn.Module or any other object (e.g.PredictorModel) that can be called with inputs and kwargs"""
        if isinstance(inputs , BatchInput) or isinstance(inputs , BatchData): 
            inputs = inputs['x']
        device0 = Device.get_device(module)
        device1 = Device.get_device(inputs)
        if device0 != device1:
            module = module.to(device1)
        outputs = module(inputs ,  **kwargs)
        batch_output = cls(outputs)
        Logger.stdout(batch_output.info)
        return batch_output

@dataclass
class BatchData:
    input : BatchInput
    output : BatchOutput
    def __len__(self): return len(self.input)
    def __getitem__(self, key) -> Any:
        if key in self.input.kwargs.keys() and key in self.output.other.keys():
            return {f'input.{key}':self.input[key] , f'output.{key}':self.output.other[key]}
        elif key in ['x' , 'y' , 'w' , 'i' , 'eff' , *self.input.kwargs.keys()]:
            return self.input[key]
        elif key in ['pred' , *self.output.other.keys()]:
            return self.output[key]
        else:
            raise KeyError(f'{key} is not a valid key')
    def keys(self):
        return list(set([*self.input.keys() , *self.output.keys()]))
    def items(self):
        return {k:self[k] for k in self.keys()}
    @property
    def shape(self):
        return self.input.shape | self.output.shape
    @property
    def info(self):
        return f'{self.__class__.__name__}:\n' + \
            f'input : {self.input.info}\n' + \
            f'output : {self.output.info}\n'
    @property
    def device(self): 
        return self.input.device

    def pred_df(self , * , narrow_df = False , colnames : Base.alias.NamesType = None , colname_prefix : str = '' , label : bool = True , **kwargs):
        df = self.output.pred_df(
            self.input.secid , self.input.date , narrow_df = narrow_df , 
            colnames = colnames , colname_prefix = colname_prefix , **kwargs
        )
        if label:
            df['label'] = self.input.label0
        return df

    def hidden_df(self , * , narrow_df = False , colnames : Base.alias.NamesType = None , colname_prefix : str = '' , **kwargs):
        df = self.output.hidden_df(
            self.input.secid , self.input.date , narrow_df = narrow_df , 
            colnames = colnames , colname_prefix = colname_prefix , **kwargs
        )
        return df

    def hidden_df_pl(self , colname_prefix : str = ''):
        """
        get hidden dataframe in polars format
        Args:
            colname_prefix : str = '', the prefix of the column names
        Returns:
            pl.DataFrame
        """
        full_hidden : torch.Tensor | Any = self.output.hidden
        full_hidden = full_hidden.cpu().numpy()

        assert full_hidden.ndim == 2 , full_hidden.shape
        columns = [f'hidden.{i}' for i in range(full_hidden.shape[-1])]
        if colname_prefix:
            columns = [f'{colname_prefix}.{col}' for col in columns]
        assert full_hidden.shape[-1] == len(columns) , (full_hidden.shape , columns)
        import polars as pl
        df = pl.DataFrame({'secid' : self.input.secid , 'date' : self.input.date , **{col:full_hidden[:,i] for i,col in enumerate(columns)}})
        return df

    def loss_inputs(self , exclude_nan : bool = False):
        pred , label , weight = self.output.pred , self.input.y , self.input.w
        if label.ndim == 1: 
            label = label[:,None]
            if weight is not None:
                weight = weight[:,None]
        if pred.ndim == 1: 
            pred  = pred[:,None]
        others = self.output.other

        row_pos = self._slice_nonnan(pred , label , weight) if exclude_nan else None
        if row_pos is not None:
            label = label[row_pos]
            pred = pred[row_pos]
            weight = weight[row_pos] if weight is not None else None

            others = {}
            for key , value in self.output.other.items():
                if (
                    not isinstance(value , torch.Tensor)
                    or value.ndim == 0
                    or value.shape[0] != len(row_pos)
                ):
                    others[key] = value
                    continue
                if value.ndim == 2 and value.shape[1] == len(value):
                    Logger.warning(f'{key} is a 2-dim square tensor of {value.shape}, remove nan for both rows and columns')
                    others[key] = value[row_pos][:,row_pos]
                else:
                    others[key] = value[row_pos]
        return {'label':label , 'pred':pred , 'weight':weight , **others}

    def _slice_nonnan(self , *args : torch.Tensor | None , print_all_nan = False) -> torch.Tensor | None:
        nanpos = False
        tensors = [arg for arg in args if arg is not None]
        if not tensors:
            return None
        nanpos = torch.zeros_like(tensors[0])
        for ts in tensors:
            nanpos += ts.isnan().any(dim = -1 , keepdim = True)
        if nanpos.ndim > 1:
            nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0 
        if print_all_nan and nanpos.all(): 
            Logger.error('Encountered all nan inputs in metric calculation!')
            [Logger.stdout(arg) for arg in args]
        return ~nanpos

    @property
    def batch_date(self) -> int:
        return self.input.date0

