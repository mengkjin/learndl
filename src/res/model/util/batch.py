import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from dataclasses import dataclass , field
from typing import Any, Literal

from src.proj import Proj , Logger
from src.proj.util import Device
from src.proj.func import properties

def _object_shape(obj : Any) -> Any:
    if obj is None: 
        return None
    elif isinstance(obj , torch.Tensor | np.ndarray): 
        return obj.shape
    elif isinstance(obj , (list , tuple)): 
        return tuple([_object_shape(x) for x in obj])
    else: 
        return type(obj)

@dataclass(slots=True)
class BatchInput:
    '''custom data component of a batch(x,y,w,i,valid)'''
    x       : torch.Tensor | tuple[torch.Tensor,...] | list[torch.Tensor]
    y       : torch.Tensor 
    w       : torch.Tensor | None
    i       : torch.Tensor 
    valid   : torch.Tensor
    kwargs  : dict[str,Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.x , (list , tuple)) and len(self.x) == 1: 
            self.x = self.x[0]
        assert self.y is not None , 'y must not be None'
        assert self.i is not None , 'i must not be None'
        assert self.valid is not None , 'valid must not be None'
        assert self.w is None or self.w.shape == self.y.shape , (self.w.shape , self.y.shape)
    def to(self , device = None): 
        if device is None: 
            return self
        else:
            if isinstance(device , Device): 
                device = device.device
            inputs = {name:Device.send_to(getattr(self , name) , device) for name in self.__slots__}
            return BatchInput(**inputs)
        
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
        elif key == 'valid': 
            return self.valid
        else: 
            return self.kwargs[key]
    def keys(self):
        return ['x' , 'y' , 'w' , 'i' , 'valid' , *self.kwargs.keys()]
    def items(self):
        return {k:self[k] for k in self.keys()}
    
    @property
    def empty(self): return len(self.y) == 0
    @property
    def device(self): return self.y.device
    @property
    def shape(self): 
        return properties.shape(self , self.keys())
    @property
    def info(self):
        return f'{self.__class__.__name__}:\n' + \
            '\n'.join([f'{k} : {shape}' for k,shape in self.shape.items()])
    @classmethod
    def random(cls , batch_size = 2 , seq_len = 30 , n_inputs = 6 , predict_steps = 1):

        x = torch.rand(batch_size , seq_len , n_inputs)

        y = torch.rand(batch_size , predict_steps)
        w = None
        i = torch.Tensor([])
        v = torch.Tensor([])
        return cls(x , y , w , i , v)
    @classmethod
    def concat(cls , *batch_datas):
        assert len(batch_datas) > 0
        
        x , y , w , i , v , kwargs = [] , [] , [] , [] , [] , []
        for bd in batch_datas:
            assert isinstance(bd , cls) , type(bd)
            x.append(bd.x)
            y.append(bd.y)
            w.append(bd.w)
            i.append(bd.i)
            v.append(bd.valid)
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
        return cls(x , y , w , i , v)

    @classmethod
    def generate(cls , data_type : str = 'day+style' , label_num : Literal[1,2] = 1):
        if data_type == 'random':
            return cls.random()
        else:
            from src.res.model.util import ModelConfig
            from src.res.model.data_module.module import DataModule

            override = {
                'model.module':'gru',
                'input.type':'data',
                'short_test':True,
                'input.data.types':data_type ,
                'model.labels': ['std_lag1_10' , 'std_lag1_20'] if label_num == 2 else ['std_lag1_10'],
                'num_output':[label_num],
            }
            with Proj.Silence:
                model_config = ModelConfig(None, override=override, test_mode=True)
                data = DataModule(model_config , 'predict').load_data()
                data.setup('predict' , model_date = data.datas.y.date[-50])
                batch_input = data.predict_dataloader()[0]
            Logger.stdout(batch_input.info)
            return batch_input
@dataclass(slots=True)
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
        return properties.shape(self , self.keys())
    @property    
    def empty(self): return len(self) == 0
    @property
    def device(self): return self.pred.device
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
    
    def pred_df(self , secid : np.ndarray | Any , date : np.ndarray | Any , narrow_df = False , 
                colnames : str | list | None = None , **kwargs):
        pred = self.pred.cpu().numpy()
        if pred.ndim == 1: 
            pred = pred[:,None]
        assert pred.ndim == 2 , pred.shape

        if colnames is None: 
            columns = [f'pred.{i}' for i in range(pred.shape[-1])]
        elif isinstance(colnames , str): 
            columns = [colnames]
        else: 
            columns = colnames
        assert pred.shape[-1] == len(columns) , (pred.shape , columns)

        df = pd.DataFrame({'secid' : secid , 'date' : date , **{col:pred[:,i] for i,col in enumerate(columns)}})
        if isinstance(colnames , str):
            assert pred.shape[-1] == 1 , (pred.shape , colnames)
            df = df.rename(columns={'pred.0':colnames})
        if narrow_df: 
            df = df.melt(['secid','date'] , var_name='feature')
        return df.assign(**kwargs)
        
    def hidden_df(self , secid : np.ndarray , date : np.ndarray , narrow_df = False ,
                  colnames : str | list | None = None , **kwargs):
        '''kwargs will be used in df.assign(**kwargs)'''
        full_hidden : torch.Tensor | Any = self.other['hidden']
        full_hidden = full_hidden.cpu().numpy()

        assert full_hidden.ndim == 2 , full_hidden.shape

        if colnames is None: 
            columns = [f'hidden.{i}' for i in range(full_hidden.shape[-1])]
        elif isinstance(colnames , str): 
            columns = [colnames]
        else: 
            columns = colnames
        assert full_hidden.shape[-1] == len(columns) , (full_hidden.shape , columns)

        df = pd.DataFrame({'secid' : secid , 'date' : date , **{col:full_hidden[:,i] for i,col in enumerate(columns)}})
        if narrow_df: 
            df = df.melt(['secid','date'] , var_name='feature')
        return df.assign(**kwargs)
    
    @classmethod
    def from_module(cls , module : nn.Module | Any , inputs : 'Any | BatchInput | BatchData' , **kwargs):
        """module can be a nn.Module or any other object (e.g.PredictorModel) that can be called with inputs and kwargs"""
        if isinstance(inputs , BatchInput) or isinstance(inputs , BatchData): 
            inputs = inputs['x']
        device0 = Device.get_device(module)
        device1 = Device.get_device(inputs)
        assert device0 == device1 , (device0 , device1)
        outputs = module(inputs ,  **kwargs)
        batch_output = cls(outputs)
        Logger.stdout(batch_output.info)
        return batch_output

@dataclass(slots=True)
class BatchData:
    input : BatchInput
    output : BatchOutput
    def __len__(self): return len(self.input)
    def __getitem__(self, key) -> Any:
        if key in self.input.kwargs.keys() and key in self.output.other.keys():
            return {f'input.{key}':self.input[key] , f'output.{key}':self.output.other[key]}
        elif key in ['x' , 'y' , 'w' , 'i' , 'valid' , *self.input.kwargs.keys()]:
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

    def loss_inputs(self , exclude_nan : bool = False):
        label , pred , weight = self.input.y , self.output.pred , self.input.w
        if label.ndim == 1: 
            label = label[:,None]
            if weight is not None:
                weight = weight[:,None]
        if pred.ndim == 1: 
            pred  = pred[:,None]
        
        others = self.output.other
        row_pos = self._slice_nonnan(label , pred , weight) if exclude_nan else None
        if row_pos is not None:
            label = label[row_pos]
            pred = pred[row_pos]
            weight = weight[row_pos] if weight is not None else None
            for key , value in others.items():
                if not isinstance(value , torch.Tensor) or len(value) != len(row_pos):
                    continue
                if value.ndim == 2 and value.shape[1] == len(value):
                    Logger.warning(f'{key} is a 2-dim square tensor of {value.shape}, remove nan for both rows and columns')
                    others[key] = others[key][row_pos][:,row_pos]
                else:
                    others[key] = others[key][row_pos]

        return {'label':label , 'pred':pred , 'weight':weight , **others}

    @staticmethod
    def _slice_nonnan(*args : torch.Tensor | None , print_all_nan = False) -> torch.Tensor | None:
        nanpos = False
        tensors = [arg for arg in args if arg is not None]
        if not tensors:
            return None
        nanpos = torch.zeros_like(tensors[0])
        for ts in tensors:
            nanpos += ts.isnan()
        if nanpos.ndim > 1:
            nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0 
        if print_all_nan and nanpos.all(): 
            Logger.error('Encountered all nan inputs in metric calculation!')
            [Logger.stdout(arg) for arg in args]
        return ~nanpos

