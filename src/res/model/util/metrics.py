import numpy as np
import pandas as pd
import torch

from torch import nn , Tensor
from typing import Any , Callable , Literal

from src.proj import Logger , Proj
from src.res.algo.nn.loss import Accuracy , Loss , MultiHeadLosses

from .batch import BatchData

class Metrics:
    '''calculator of batch output'''
    VAL_DATASET : Literal['train','valid'] = 'valid'
    VAL_METRIC  : Literal['loss' ,'accuracy'] = 'accuracy'
    
    def __init__(self , 
                 module_type = 'nn' , nn_category = None ,
                 criterion_loss : dict[str,dict[str,Any]] = {'mse':{}} , 
                 criterion_accuracy : dict[str,dict[str,Any]] = {'spearman':{}} , 
                 criterion_multilosses : dict[str,Any] = {} ,
                 **kwargs) -> None:
        assert len(criterion_loss) > 0 , f'{criterion_loss} should be not empty'
        assert len(criterion_accuracy) > 0 , f'{criterion_accuracy} should be only one'
        
        self.module_type = module_type
        self.nn_category = nn_category
        self.criterion_loss = criterion_loss
        self.criterion_accuracy = criterion_accuracy
        self.criterion_multilosses = criterion_multilosses

        self.batch_metrics = BatchMetrics()
        self.epoch_metrics = EpochMetrics()
        self.attempt_metrics = AttemptMetrics()
        self.model_metrics = ModelMetrics()

    def __repr__(self):
        return f'{self.__class__.__name__}(loss={self.criterion_loss},metric={self.criterion_accuracy})'

    @property
    def batch_accuracy(self): 
        return self.batch_metrics.total_accuracy
    @property
    def batch_loss(self): 
        return self.batch_metrics.total_loss_item
    @property
    def epoch_batch_accuracies(self): 
        return self.epoch_metrics.averages['accuracies']
    @property
    def epoch_batch_keys(self): 
        return np.array(self.epoch_metrics.batch_index)
    @property
    def epoch_losses(self) -> dict:
        if not self.epoch_metrics.tables['losses']:
            return {}
        df = self.epoch_metrics.tables['losses'][-1]
        return df.iloc[-1].to_dict()
    @property
    def epoch_accuracies(self) -> dict:
        if not self.epoch_metrics.tables['accuracies']:
            return {}
        df = self.epoch_metrics.tables['accuracies'][-1]
        return df.iloc[-1].to_dict()
    @property
    def last_epoch_metric(self) -> float:
        return self.attempt_metrics.latest(self.dataset , self.VAL_METRIC)
    @property
    def best_epoch_metric(self) -> float | None:
        return self.attempt_metrics.best(self.dataset , self.VAL_METRIC)
    @property
    def multilosses_kwargs(self): 
        return self.criterion_multilosses | {'num_head':self.model_param.get('num_output' , 1)}
    @property
    def which_output(self) -> int | list[int] | None: 
        return self.model_param.get('which_output' , None)
    @property
    def which_label(self) -> int | list[int] | None: 
        return self.model_param.get('which_label' , None)
    @property
    def ignore_loss(self) -> list[str]:
        ignores = []
        if (self.nn_category == 'tra') or self.model_param.get('hidden_as_factors' , False):
            ignores.extend(['hidden_corr' , 'hidden_corr_deprecated'])
        return ignores
    @property
    def ignore_accuracy(self) -> list[str]:
        return []

    def calculate(self , dataset : Literal['train','valid','test','predict'] , batch_key : Any , batch_data : BatchData):
        '''Calculate loss(with gradient), penalty , accuracy'''
        self.new_batch(batch_key = batch_key)

        accuracies = self.accuracy_function.accuracies(batch_data, self.which_output , self.which_label , require_grad = False)
        if self.module_type == 'nn' and dataset in ['train','valid']:
            losses = self.loss_function.losses(batch_data , self.which_output , self.which_label , require_grad = dataset == 'train')
        else:
            losses = {}

        self.batch_metrics.set_values(accuracies , losses)
        self.collect_batch()
        return self

    def new_all(self , model : nn.Module | Any , model_param : dict[str,Any] , **kwargs):
        self.new_model(model , model_param , **kwargs)
        self.new_attempt(**kwargs)
        self.new_epoch(**kwargs)
        return self

    def new_model(self , model : nn.Module | Any , model_param : dict[str,Any] , **kwargs):
        self.model_param  = model_param
        if isinstance(model , nn.Module):
            net = model
        elif hasattr(model , 'net') and [cls.__qualname__ for cls in Proj.States.trainer.model.__class__.__mro__]:
            net = getattr(model , 'net')
            assert net is None or isinstance(net , nn.Module) , f'{net} is not a torch.nn.Module'
        else:
            net = None
        self.loss_function = LossFunction(self.criterion_loss , net , self.ignore_loss , self.multilosses_kwargs)
        self.accuracy_function = AccuracyFunction(self.criterion_accuracy , net , self.ignore_accuracy)
        self.model_metrics.new(**kwargs)
        return self

    def new_attempt(self , attempt : int = 0 , **kwargs):
        self.attempt_metrics.new(attempt = attempt , **kwargs)

    def new_epoch(self , dataset : Literal['train','valid','test'] , epoch : int , **kwargs):
        self.dataset : Literal['train','valid','test'] = dataset
        self.epoch_metrics.new(self.dataset , epoch , **kwargs)

    def new_batch(self , batch_key : Any , **kwargs):
        self.batch_metrics.new(batch_key = batch_key , **kwargs)

    def collect_batch(self):
        self.epoch_metrics.append(self.batch_metrics)
        self.batch_metrics.close()

    def collect_epoch(self):
        self.attempt_metrics.append(self.epoch_metrics)
        self.epoch_metrics.close()

    def collect_attempt(self):
        self.model_metrics.append(self.attempt_metrics)
        self.attempt_metrics.close()

    def collect_model(self):
        self.model_metrics.close()

    def better_epoch(self , old_best_epoch : Any) -> bool:
        if old_best_epoch is None:
            return True
        last_metric = self.last_epoch_metric
        return last_metric > old_best_epoch if self.VAL_METRIC == 'accuracy' else last_metric < old_best_epoch
        
    def better_attempt(self , old_best_attempt : Any) -> bool:
        if old_best_attempt is None:
            return True
        best_metric = self.best_epoch_metric
        return best_metric > old_best_attempt if self.VAL_METRIC == 'accuracy' else best_metric < old_best_attempt

class GradientMode:
    def __init__(self , require_grad : bool = True):
        self.require_grad = require_grad

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(self.require_grad)

    def __exit__(self , exc_type , exc_value , traceback):
        torch.set_grad_enabled(self.prev)

class MetricFunction:
    DISPLAY_LOG : dict[str,bool] = {}
    SearchList : list[str] = []
    ExcludeNan : bool = True
    def __init__(
        self , 
        criterions : dict[str,dict[str,Any]] , 
        net : nn.Module | None = None , 
        ignores : list[str] | None = None ,
    ) -> None:
        self.criterions = criterions
        self.net = net
        self.ignores = ignores or []
        self.components : dict[str,MetricComponent] = {}
 
    def __repr__(self):
        return f'{self.__class__.__name__}(components={{{', '.join([f"{key}: {value.lamb}" for key, value in self.components.items()])}}})'

    def __call__(
        self , data : BatchData , 
        which_output : int | list[int] | None = None , 
        which_label : int | list[int] | None = None , 
        require_grad : bool = True
    ) -> dict[str,Tensor]:
        with GradientMode(require_grad):
            inputs = data.loss_inputs(exclude_nan = self.ExcludeNan)
            results : dict[str,Tensor] = {}
            for criterion , component in self.components.items():
                if not self.DISPLAY_LOG.get(criterion , False):
                    Logger.success(f'{self.__class__.__name__} {criterion} calculated!' , vb_level = 'max')
                    self.DISPLAY_LOG[criterion] = True
                loss = component(which_output = which_output , which_label = which_label , **inputs)
                if isinstance(loss , dict):
                    results.update(loss)
                else:
                    results[criterion] = loss
        return results

class LossFunction(MetricFunction):
    DISPLAY_LOG : dict[str,bool] = {}
    SearchList : list[str] = ['loss' , 'Loss' , 'loss_function']
    ExcludeNan : bool = False
    def __init__(
        self , 
        criterions : dict[str,dict[str,Any]] , 
        net : nn.Module | None = None , 
        ignores : list[str] | None = None ,
        multilosses_kwargs : dict[str,Any] | None = None ,
    ) -> None:
        super().__init__(criterions , net , ignores)
        self.multilosses_kwargs = multilosses_kwargs or {}
        self.init_components()

    def init_components(self):
        self.components : dict[str,LossComponent] = {}
        if self.net is not None:
            for name in self.SearchList:
                if hasattr(self.net , name):
                    calculator : Any = getattr(self.net , name)
                    self.components['net_specific'] = LossComponent(calculator)
                    return self

        for i , (criterion , kwargs) in enumerate(self.criterions.items()):
            if criterion in self.ignores:
                continue
            calculator = Loss.get(criterion , **kwargs)
            if i == 0 and calculator.multiheadlosses_capable:
                multilosses = MultiHeadLosses(**self.multilosses_kwargs , mt_param = MultiHeadLosses.get_params(self.net))
            else:
                multilosses = None
            self.components[criterion] = LossComponent(calculator , multilosses = multilosses , **kwargs)

    def losses(self , data : BatchData , which_output : int | list[int] | None = None , which_label : int | list[int] | None = None ,
               prefix : str | tuple[str,...] | None = ('penalty_' , 'loss_') , require_grad : bool = True) -> dict[str,Tensor]:
        losses = self(data, which_output = which_output , which_label = which_label , require_grad = require_grad)
        if prefix is not None:
            losses.update({key:value for key , value in data.output.other.items() if key.lower().startswith(prefix)})
        return losses

class AccuracyFunction(MetricFunction):
    DISPLAY_LOG : dict[str,bool] = {}
    SearchList : list[str] = []
    ExcludeNan : bool = True
    def __init__(
        self , 
        criterions : dict[str,dict[str,Any]] , 
        net : nn.Module | None = None , 
        ignores : list[str] | None = None ,
    ) -> None:
        super().__init__(criterions , net , ignores)
        self.init_components()

    def init_components(self):
        self.components : dict[str,AccuracyComponent] = {}
        if self.net is not None:
            for name in self.SearchList:
                if hasattr(self.net , name):
                    self.components['net_specific'] = AccuracyComponent(getattr(self.net , name))
                    return self

        for i , (criterion , kwargs) in enumerate(self.criterions.items()):
            if criterion in self.ignores:
                continue
            calculator = Accuracy.get(criterion , **kwargs)
            self.components[criterion] = AccuracyComponent(calculator , **kwargs)

    def accuracies(self , data : BatchData , which_output : int | list[int] | None = None , which_label : int | list[int] | None = None , require_grad : bool = False) -> dict[str,float]:
        accuracies = self(data, which_output = which_output , which_label = which_label , require_grad = require_grad)
        accuracies = {key:value.item() if isinstance(value , Tensor) else value for key,value in accuracies.items()}
        return accuracies

    def hidden_accuracies(self , data : BatchData , which_label : int = 0 , require_grad : bool = False) -> torch.Tensor:
        assert not require_grad , 'hidden accuracies do not support require_grad'
        hidden = data.output.hidden
        label = data.input.y[...,which_label:which_label+1]
        component = self.components[list(self.components.keys())[0]]
        with GradientMode(require_grad):
            accuracies = component(pred = hidden , label = label)
        if isinstance(accuracies , dict):
            assert len(accuracies) == 1 , f'hidden accuracies should be a single value, but got {accuracies}'
            accuracies = list(accuracies.values())[0]
        return accuracies

class MetricComponent:
    def __init__(
        self , 
        calculator : Callable[...,Tensor | dict[str,Tensor]] , 
        lamb : float = 1. , 
        dim : int | None = 0 ,
        which_output : int | list[int] | None = None ,
        which_label : int | list[int] | None = None ,
        **kwargs
    ):
        self.calculator = calculator
        self.lamb = lamb
        self.dim = dim
        self.which_output = which_output
        self.which_label = which_label
        self.kwargs = kwargs

    def __call__(self , **kwargs) -> Tensor | dict[str,Tensor]:
        kwargs = self.filter_inputs(**kwargs)
        output = self.calculator(dim = self.dim , **self.kwargs , **kwargs)
        return self.apply_lamb(output)

    def __repr__(self):
        return f'{self.__class__.__name__}(calculator={self.calculator},lamb={self.lamb},dim={self.dim},which_output={self.which_output},which_label={self.which_label},kwargs={self.kwargs})'

    def apply_lamb(self , output : Tensor | dict[str,Tensor]) -> Tensor | dict[str,Tensor]:
        if isinstance(output , dict):
            return {k:self.lamb * v for k,v in output.items()}
        else:
            return self.lamb * output

    def filter_inputs(self , which_output : int | list[int] | None = None , which_label : int | list[int] | None = None , **kwargs):
        label , pred , weight = kwargs.get('label' , None) , kwargs.get('pred' , None) , kwargs.get('weight' , None)
        which_output = which_output or self.which_output
        which_label = which_label or self.which_label
        if which_output is not None:
            if pred is not None:
                pred = pred[...,which_output]
            if weight is not None:
                weight = weight[...,which_output]
        if which_label is not None:
            if label is not None:
                label = label[...,which_label]
        if pred is not None and pred.ndim == 1:
            pred = pred[:,None]
        if label is not None and label.ndim == 1:
            label = label[:,None]
        if weight is not None and weight.ndim == 1:
            weight = weight[:,None]
        return kwargs | {'label':label , 'pred':pred , 'weight':weight}

class LossComponent(MetricComponent):
    def __init__(
        self , 
        calculator : Callable[...,Tensor | dict[str,Tensor]] , 
        lamb : float = 1. , 
        dim : int | None = 0 ,
        which_output : int | list[int] | None = None ,
        which_label : int | list[int] | None = None ,
        multilosses : MultiHeadLosses | None = None ,
        **kwargs
    ):
        super().__init__(calculator , lamb , dim , which_output , which_label , **kwargs)
        self.multilosses = multilosses

    def __call__(self , **kwargs) -> Tensor | dict[str,Tensor]:
        output = super().__call__(**kwargs)
        if self.multilosses:
            assert isinstance(output , Tensor) , f'loss output should be a Tensor when multilosses is applied, but got {output}'
            output = self.multilosses(output , mt_param = kwargs.get('mt_param' , {}))
        return output
class AccuracyComponent(MetricComponent):
    def __call__(self , **kwargs) -> Tensor | dict[str,Tensor]:
        output = super().__call__(**kwargs)
        return output

class BatchMetrics:
    def __init__(self) -> None:
        self.key = {}
        self.initiated = False
        self.collected = False
        self.accuracies : dict[str,float] = {}
        self.losses : dict[str,Tensor] = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key},accuracies={self.accuracies},losses={self.losses})'

    def new(self , batch_key : Any = None , **kwargs):
        assert not self.initiated , f'{self} is already initialized , please call close() first'
        self.initiated = True
        self.collected = False
        self.key = {'batch':batch_key}

    def set_values(self , accuracies : dict[str,float] , losses : dict[str,Tensor]):
        assert self.initiated , 'BatchMetrics is not initialized , please call new(batch_key) first'
        self.accuracies = accuracies
        self.losses = losses
        self._total_loss = None
        self._total_accuracy = None

    def close(self):
        assert self.collected , f'{self} is not collected before closing, please be appended to some metrics first'
        self.initiated = False

    def empty_metrics(self , name : str) -> bool:
        if name == 'accuracies':
            return not bool(self.accuracies)
        elif name == 'losses':
            return not bool(self.losses)
        else:
            raise ValueError(f'Invalid metric name: {name}')

    @property
    def total_loss(self) -> Tensor:
        if not self.losses:
            return torch.Tensor([0.])
        if not hasattr(self , '_total_loss') or self._total_loss is None:
            losses = torch.concatenate([value.flatten() for value in self.losses.values()])
            self._total_loss = losses.sum() if losses.numel() > 0 else torch.Tensor([0]).requires_grad_(True)
        return self._total_loss

    @property
    def total_accuracy(self) -> float:
        if not self.accuracies:
            return np.nan
        if not hasattr(self , '_total_accuracy') or self._total_accuracy is None:
            accuracies = [value for value in self.accuracies.values()]
            self._total_accuracy = sum(accuracies) if accuracies else 0.
        return self._total_accuracy

    @property
    def total_loss_item(self) -> float: 
        return self.total_loss.item()

    @property
    def accuracies_items(self) -> dict[str,float]: 
        return self.accuracies

    @property
    def losses_items(self) -> dict[str,float]:
        return {key:value.item() for key,value in self.losses.items()}

    @property
    def table_accuracies(self) -> pd.DataFrame:
        return pd.DataFrame(self.accuracies_items , index = pd.MultiIndex.from_frame(pd.DataFrame(self.key , index = [0])))
    @property
    def table_losses(self) -> pd.DataFrame:
        return pd.DataFrame(self.losses_items , index = pd.MultiIndex.from_frame(pd.DataFrame(self.key , index = [0])))

    @property
    def batch_key(self) -> Any:
        return self.key['batch']

class AggregatedMetrics:
    def __init__(self , metric_names : list[str] = ['accuracies' , 'losses']) -> None:
        self.key = {}
        self.metric_names = metric_names
        self.initiated = False
        self.collected = False
        self.tables : dict[str,list[pd.DataFrame]] = {name:[] for name in metric_names}
        self.averages : dict[str,list[float]] = {name:[] for name in metric_names}
    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key},metrics={self.metric_names})'
    def new(self , **kwargs):
        assert not self.initiated , f'{self} is already initialized , please call close() first'
        self.initiated = True
        self.collected = False
        self.key = kwargs
        for name in self.metric_names:
            self.tables[name].clear()
            self.averages[name].clear()
    def append(self , metrics : 'BatchMetrics | AggregatedMetrics'):
        assert self.initiated , f'{self} is not initialized , please call new() first'
        assert not metrics.collected , f'{metrics} is already collected , please call new() first'
        metrics.collected = True
        # define specific append method for different levels
        
    def close(self):
        assert self.collected , f'{self} is not collected before closing, please be appended to some metrics first'
        self.initiated = False

    def get_table(self , name : str) -> pd.DataFrame:
        if not self.tables[name]:
            return pd.DataFrame()
        df = pd.concat(self.tables[name])
        cols = [*self.key.keys() , *df.index.names]
        return df.assign(**self.key).reset_index(drop = False).set_index(cols)

    def get_average(self , name : str) -> float:
        if len(self.averages[name]) == 0:
            return np.nan
        else:
            return np.mean(self.averages[name]).item()

    def empty_metrics(self , name : str) -> bool:
        return not bool(self.tables[name])

class EpochMetrics(AggregatedMetrics):
    '''record a list of batch metric and perform agg operations, usually used in an epoch'''
    def __init__(self):
        super().__init__(['accuracies' , 'losses'])
        self.batch_index = []

    def new(self , dataset , epoch = 0 , **kwargs):
        super().new(dataset = dataset , epoch = epoch)
        self.batch_index.clear()
    
    def append(self , metrics : BatchMetrics):
        super().append(metrics)
        self.batch_index.append(metrics.batch_key)
        self.tables['accuracies'].append(metrics.table_accuracies)
        self.tables['losses'].append(metrics.table_losses)
        self.averages['accuracies'].append(metrics.total_accuracy)
        self.averages['losses'].append(metrics.total_loss_item)

    @property
    def dataset(self):
        return self.key['dataset']
    @property
    def epoch(self):
        return self.key['epoch']
    @property
    def nanloss(self): 
        return np.isnan(self.get_average('losses')).any()
        
class AttemptMetrics(AggregatedMetrics):
    '''record a list of dataset metric and perform agg operations, usually used in an attempt'''
    def __init__(self):
        super().__init__(['train_accuracies' , 'valid_accuracies' , 'test_accuracies' , 'train_losses' , 'valid_losses' , 'test_losses'])

    def new(self , attempt : int = 0 , **kwargs):
        super().new(attempt = attempt)

    def append(self , metrics : EpochMetrics):
        super().append(metrics)
        for metric in ['accuracies' , 'losses']:
            if metrics.empty_metrics(metric):
                continue
            self.tables[f'{metrics.dataset}_{metric}'].append(metrics.get_table(metric))
            self.averages[f'{metrics.dataset}_{metric}'].append(metrics.get_average(metric))

    def latest(self , dataset : Literal['train','valid','test'] , metric : Literal['loss','accuracy']) -> float:
        metric_list = self.averages[f'{dataset}_losses'] if metric == 'loss' else self.averages[f'{dataset}_accuracies']
        return metric_list[-1] if metric_list else 0.

    def best(self , dataset : Literal['train','valid','test'] , metric : Literal['loss','accuracy']) -> float | None:
        metric_list = self.averages[f'{dataset}_losses'] if metric == 'loss' else self.averages[f'{dataset}_accuracies']
        if len(metric_list) == 0:
            return None
        return min(metric_list) if metric == 'loss' else max(metric_list)

class ModelMetrics(AggregatedMetrics):
    def __init__(self):
        super().__init__(['train_accuracies' , 'valid_accuracies' , 'test_accuracies' , 'train_losses' , 'valid_losses' , 'test_losses'])

    def new(self , stage : Literal['fit' , 'test' , 'predict'] = 'fit' , model_num : int = 0 , model_date : int = 0 , submodel : str  = 'best' , **kwargs):
        super().new(stage = stage , model_num = model_num , model_date = model_date , submodel = submodel)

    def append(self , metrics : AttemptMetrics):
        super().append(metrics)
        for name in self.metric_names:
            if metrics.empty_metrics(name):
                continue
            self.tables[name].append(metrics.get_table(name))
            self.averages[name].append(metrics.get_average(name))

    def close(self):
        # self.export_metrics()
        self.collected = True
        super().close()

    def export_metrics(self):
        dfs = {name : self.get_table(name) for name in self.metric_names}
        dfs = {name : df.reset_index(drop = False).drop(columns = ['stage','model_num','model_date','submodel','dataset']) 
               for name , df in dfs.items() if not df.empty}
        if dfs:
            def summarize(x : pd.DataFrame , name : str , **kwargs):
                return pd.Series({f'{name}_batches' : x.shape[0] , f'{name}' : x.sum(1).mean()})
            summaries = [df.set_index(['attempt' , 'epoch' , 'batch']).groupby(['attempt' , 'epoch']).apply(summarize , name = name , include_groups = False) for name , df in dfs.items()] 
            summary = pd.concat(summaries , axis = 1).drop(columns = ['train_accuracies_batches' , 'valid_accuracies_batches']).\
                rename(columns = {'train_accuracies':'train_accuracy' , 'valid_accuracies':'valid_accuracy' , 'train_losses':'train_loss' , 
                'valid_losses':'valid_loss' ,'train_losses_batches':'train_batches' , 'valid_losses_batches':'valid_batches'})
            columns = summary.columns.intersection(['train_batches' , 'train_loss' , 'train_accuracy' , 'valid_batches' , 'valid_loss' , 'valid_accuracy'])
            summary = summary.loc[:,columns].reset_index(drop = False)
            for attempt in summary['attempt'].unique():
                dfs[f'attempt_{attempt}'] = summary.query('attempt == @attempt').drop(columns = ['attempt'])
        return dfs

    @property
    def model_name(self) -> str:
        return f'{self.key["model_num"]}.{self.key["model_date"]}.xlsx'
