"""
Callback to summarize information of trainer
- SummaryWriter : write summary to tensorboard and json file
"""
from __future__ import annotations
import torch
import shutil


from datetime import datetime
from functools import cached_property
from typing import Any
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TsboardWriter

from src.proj import Proj , PATH , Save , Base
from src.res.model.util import BaseCallBack

__all__ = ['SummaryWriter']

def _hidden_correlation_to_chw(hcorr : torch.Tensor) -> torch.Tensor:
    hcorr = hcorr.clamp(-1,1)
    r, g, b = torch.ones_like(hcorr), torch.ones_like(hcorr), torch.ones_like(hcorr)
    pos_mask = hcorr > 0
    g[pos_mask] = 1.0 - hcorr[pos_mask]
    b[pos_mask] = 1.0 - hcorr[pos_mask]
    neg_mask = hcorr < 0
    abs_hidden_correlation = hcorr.abs()
    r[neg_mask] = 1.0 - abs_hidden_correlation[neg_mask]
    b[neg_mask] = 1.0 - abs_hidden_correlation[neg_mask]
    return torch.stack([r, g, b], dim=0)

def _format_messages(messages : dict[str, str | dict] , indent : int = 0) -> list[str]:
    key_len = max(len(key) for key in messages.keys())
    msgs : list[str] = []
    for key , msg in messages.items():
        if isinstance(msg , dict):
            msgs.append('  ' * indent + f'{key:{key_len}s} : ')
            msgs.extend(_format_messages(msg , indent + 1))
        else:
            msgs.append('  ' * indent + f'{key:{key_len}s} : {msg}')
    return msgs

class SummaryWriter(BaseCallBack):
    """Tensorboard and Summary Writer"""
    DEBUG_STEP : int = 100 # record debug mode every step in batches
    HIDDEN_FEATURE_MODE : bool = True # enable hidden feature mode to record hidden features stats (accuracy and correlation)
    HIDDEN_FEATURE_STEP : int = 5 # record hidden features every step in epoches
    SUMMARY_NDIGITS : dict[str,int] = {'Avg':4,'Sum':2,'Std':4,'T':2,'IR':4}
    TSBOARD_PREFIXIS : dict[str,str] = {
        'metrics' : '00.Metrics',
        'events' : '01.FittingEvents',
        'hidden' : '02.HiddenFeatures',
        'hyperparameter' : '03.HyperParameter',
        'weights' : '04.ModelWeights',
        'gradients' : '05.ModelGradients',
    }

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.init_time = datetime.now()
        if self.HIDDEN_FEATURE_MODE:
            self.hidden_rankics : list[torch.Tensor] = []
            self.hidden_correlations : list[torch.Tensor] = []
        
    def __repr__(self):
        return f'{self.__class__.__name__}(trainer={self.trainer})'

    @property
    def summary_log_file(self):
        log_name = 'st_results' if self.config.base_path.is_short_test else 'results'
        from src.proj.log.logfile import LogFile
        return LogFile.initialize('model' , 'summary' , log_name)

    @property
    def last_lr(self) -> float |Any:
        if hasattr(self.model , 'optimizer'):
            return self.model.optimizer.last_lr
        return 0.

    @property
    def epoch(self) -> int:
        return self.status.model_epoch

    @property
    def batch_idx(self) -> int:
        return self.trainer.batch_idx

    @cached_property
    def writer(self):
        model_key = f'{self.config.base_path.model_clean_name}.{self.model_num}.{self.model_date}.{self.status.attempt_key}'
        return TsboardWriter(self.base_path.snapshot('tensorboard' , model_key))

    @property
    def step_epoch(self) -> int:
        return self.epoch

    @property
    def step_batch(self) -> int:
        return self.epoch * self.trainer.batch_num + self.batch_idx

    def named_parameters(self) -> list[tuple[str,nn.Parameter]]:
        net = self.model.net
        if not isinstance(net , nn.Module):
            return []
        return [(name , param) for name , param in net.named_parameters()]

    def reset_writer(self):
        if 'writer' in self.__dict__:
            del self.writer

    def add_metrics(self):
        prefix = self.TSBOARD_PREFIXIS['metrics']
        if self.status.dataset == 'train':
            accuracies = self.metrics.epoch_train_metrics.accuracies_dict
            losses = self.metrics.epoch_train_metrics.losses_dict
        else:
            accuracies = self.metrics.epoch_valid_metrics.accuracies_dict
            losses = self.metrics.epoch_valid_metrics.losses_dict
        if self.status.phase > 0:
            accuracies = {f'Phase{self.status.phase}.{key}':value for key,value in accuracies.items()}
            losses = {f'Phase{self.status.phase}.{key}':value for key,value in losses.items()}
        self.writer.add_scalars(f'{prefix}/Accuracy/{self.status.dataset.title()}' , accuracies , self.step_epoch)
        self.writer.add_scalars(f'{prefix}/Loss/{self.status.dataset.title()}' , losses , self.step_epoch)

    def add_total_metric(self):
        prefix = self.TSBOARD_PREFIXIS['metrics']
        def get_metric(ds , metric):
            return self.metrics.attempt_metrics.latest(ds , metric)
        for metric in ['accuracy' , 'loss' , 'rankic']:
            data = {'train':get_metric('train' , metric), 'valid':get_metric('valid' , metric)}
            self.writer.add_scalars(f'{prefix}/Total{metric.title()}' , data , self.step_epoch)

    def add_epoch_events(self):
        prefix = self.TSBOARD_PREFIXIS['events']
        if self.status.current.events:
            for event in self.status.current.events:
                self.writer.add_text(f'{prefix}/EventLog' , event.info , self.step_epoch)
            self.writer.add_scalar(f'{prefix}/Marker' , 1 , self.step_epoch)
        else:
            self.writer.add_scalar(f'{prefix}/Marker' , 0 , self.step_epoch)
        
    def add_hidden_rankic(self , hidden_rankics : torch.Tensor):
        prefix = self.TSBOARD_PREFIXIS['hidden']
        self.writer.add_histogram(f'{prefix}/Rankic/{self.status.dataset.title()}' , hidden_rankics , self.step_epoch)

    def add_hidden_correlation(self , hidden_correlation : torch.Tensor):
        prefix = self.TSBOARD_PREFIXIS['hidden']
        corr_heatmap = _hidden_correlation_to_chw(hidden_correlation)
        self.writer.add_image(f'{prefix}/Correlation/{self.status.dataset.title()}' , corr_heatmap , self.step_epoch , dataformats='CHW')

    def add_lr(self):
        assert self.status.dataset == 'train' , 'lr is only supported for train dataset'
        prefix = self.TSBOARD_PREFIXIS['hyperparameter']
        self.writer.add_scalar(f'{prefix}/LearnRate' , self.last_lr , self.step_epoch)
        
    def add_batch_weight_norm(self):
        assert Proj.debug and self.status.dataset == 'train' , \
            f'debug_mode is not enabled or dataset is not train : {Proj.debug} {self.status.dataset}'
        
        if not Proj.debug['tsboard_weight_norm']:
            return

        prefix = self.TSBOARD_PREFIXIS['weights']
        [self.writer.add_scalar(f'{prefix}/Norm/{name}' , torch.norm(param.data , p=2) , self.step_batch) for name , param in self.named_parameters() if param.data is not None]

    def add_batch_weight_histogram(self):
        assert Proj.debug and self.status.dataset == 'train' , \
            f'debug_mode is not enabled or dataset is not train : {Proj.debug} {self.status.dataset}'
        if not Proj.debug['tsboard_weight_histogram']:
            return
        prefix = self.TSBOARD_PREFIXIS['weights']
        [self.writer.add_histogram(f'{prefix}/Histogram/{name}' , param.data , self.step_batch) for name , param in self.named_parameters() if param.data is not None]

    def add_batch_grad_norm(self):
        assert Proj.debug and self.status.dataset == 'train' , \
            f'debug_mode is not enabled or dataset is not train : {Proj.debug} {self.status.dataset}'
        if not Proj.debug['tsboard_grad_norm']:
            return
        prefix = self.TSBOARD_PREFIXIS['gradients']
        [self.writer.add_scalar(f'{prefix}/Norm/{name}' , torch.norm(param.grad , p=2) , self.step_batch) for name , param in self.named_parameters() if param.grad is not None]

    def add_batch_grad_histogram(self):
        assert Proj.debug and self.status.dataset == 'train' , \
            f'debug_mode is not enabled or dataset is not train : {Proj.debug} {self.status.dataset}'
        if not Proj.debug['tsboard_grad_histogram']:
            return
        prefix = self.TSBOARD_PREFIXIS['gradients']
        [self.writer.add_histogram(f'{prefix}/Histogram/{name}' , param.grad , self.step_batch) for name , param in self.named_parameters() if param.grad is not None]

    def pack_tensorboard_dir(self):
        # overwrite current run folder
        ts_folder = self.config.base_path.snapshot('tensorboard')
        if not ts_folder.exists() or not any(ts_folder.iterdir()):
            return
        run_folder = PATH.tensorboard.joinpath('run')
        shutil.rmtree(run_folder , ignore_errors=True)
        shutil.copytree(ts_folder, run_folder)

        # pack run folder to tar file
        tar_filename = PATH.tensorboard.joinpath(f'{self.base_path.full_name}_{self.init_time.strftime("%Y%m%d%H%m")}.tar')
        Save.pack(
            ts_folder, tar_filename , overwrite = True , async_save = True ,
            prefix = f'{self.__class__.__name__} Tensorboard Dir' , 
            indent = self.indent + 1 , vb_level = self.vb_level + 1)

    def append_batch_hidden_summary(self):
        if self.HIDDEN_FEATURE_MODE and 'hidden' in self.batch_data.output.other:
            self.hidden_rankics.append(self.metrics.rankic_function.hidden_rankic(self.batch_data))
            self.hidden_correlations.append(torch.corrcoef(self.batch_data.output.hidden.T))

    def add_hidden_summaries(self):
        if self.HIDDEN_FEATURE_MODE:
            if self.hidden_rankics:
                hidden_rankics = torch.stack(self.hidden_rankics , -1).mean(dim=-1)
                self.hidden_rankics.clear()
                self.add_hidden_rankic(hidden_rankics)

            if self.hidden_correlations:
                hidden_correlations = torch.stack(self.hidden_correlations , -1).mean(dim=-1)
                self.hidden_correlations.clear()
                self.add_hidden_correlation(hidden_correlations)

    def on_new_attempt(self):
        self.reset_writer()

    def on_before_clip_gradients(self):
        if Proj.debug and self.batch_idx % self.DEBUG_STEP == 0:
            self.add_batch_weight_histogram()
            self.add_batch_grad_histogram()
            self.add_batch_weight_norm()
            self.add_batch_grad_norm()

    def on_train_epoch_start(self):
        self.add_lr()

    def on_train_batch_end(self):
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.append_batch_hidden_summary()

    def on_train_epoch_end(self):
        self.add_metrics()
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.add_hidden_summaries()

    def on_validation_batch_end(self):
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.append_batch_hidden_summary()

    def on_validation_epoch_end(self):
        self.add_metrics()
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.add_hidden_summaries()

    def on_fit_epoch_end(self):
        self.add_total_metric()
        self.add_epoch_events()

    def on_fit_model_end(self):
        if self.texts.model_summary:
            self.writer.add_text('Model Info' , self.texts.model_summary)

    def on_fit_end_after(self):
        self.pack_tensorboard_dir()

    def on_summarize_model(self):
        """pack tensorboard dir and export test summary to json"""
        test_summary = self.container.dataframes['test_summary']
        if test_summary.empty: 
            return
        
        test_name = f'{self.config.base_path.full_name}'

        duration : dict[str,str] = {
            stage : f'{Base.Duration(self.status.times[f'{stage}_end'] - self.status.times[f'{stage}_start'])}' for stage in self.config.queue_of_stages
        }
        metrics : dict[str,str] = {}
        for col in test_summary.columns:
            series = test_summary[col]
            key = '{}.{}'.format(*col)
            value = '|'.join([f'{k}({series.round(v).loc[k]})' for k,v in self.SUMMARY_NDIGITS.items() if k in series.index])
            metrics[key] = value

        messages = {
            f'model' : f'{self.config.model_name} x {len(self.config.model_num_list)})',
            f'submodel' : f'{self.config.submodels}',
            f'start' : f'{self.init_time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'stages' : f'{self.config.queue_of_stages}',
            f'inputs' : self.data.input_keys_subkeys,
            f'labels' : f'{self.config.labels}',
            f'range' : f'{self.config.beg_date} - {self.config.end_date}',
            f'duration' : duration,
            f'metrics' : metrics,
        }
        msgs = _format_messages(messages , indent = 0)
        self.summary_log_file.write(test_name , *msgs)
        self.logger.note(f'Summary of model {test_name} is saved to {self.summary_log_file.current_file.relative_to(PATH.main)}')
    