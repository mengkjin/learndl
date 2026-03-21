import numpy as np
import pandas as pd
import torch
import tarfile , shutil

from torch import nn
from datetime import datetime
from typing import Any , ClassVar , Literal

from src.proj import Logger , LogFile , Duration , Proj , PATH
from src.res.model.data_module import BatchInputLoader
from src.res.model.util import BaseCallBack

class CallbackTimer(BaseCallBack):
    '''record time cost of callback hooks'''
    WITH_CB = True
    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.recording = Proj.vb.is_max_level
        self.record_hook_durations : dict[str,list[float]]  = {hook:[] for hook in self.possible_hooks}
        self.record_start_time : dict[str,datetime] = {}
    def __bool__(self):
        """disable callback timer"""
        return False # self.recording and not self.turn_off
    def at_enter(self , hook_name , vb_level : Any = 'max'):
        super().at_enter(hook_name , vb_level)
        if self.recording: 
            self.record_start_time[hook_name] = datetime.now()
    def at_exit(self, hook_name , vb_level : Any = 'max'):
        if self.recording: 
            self.record_hook_durations[hook_name].append((datetime.now() - self.record_start_time[hook_name]).total_seconds())
        super().at_exit(hook_name , vb_level)
    def on_summarize_model(self):
        if self.recording: 
            columns = ['hook_name' , 'num_calls', 'total_time' , 'avg_time']
            values  = [[k , len(v) , np.sum(v) , np.mean(v)] for k,v in self.record_hook_durations.items() if v]
            df = pd.DataFrame(values , columns = columns).sort_values(by=['total_time'],ascending=False).head(5)
            Logger.display(df , caption = 'Table: Callback Time Costs:')  
            
class StatusDisplay(BaseCallBack):
    '''display epoch and event information'''
    CB_ORDER : int = 100

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.show_info_step = [0,10,5,3,2,1][min(Proj.vb.vb // 2 , 5)]
        self.dataloader_info = Proj.vb.is_max_level
        self.record_texts : dict[str,str]   = {}
        self.record_epoch_model : int = 0
        self.record_epoch_stage : int = 0
        self.record_model_stage : int = 0

    @property
    def dataloader(self) -> BatchInputLoader | Any : return self.trainer.dataloader
    @property
    def initial_models(self) -> bool: return self.record_model_stage <= self.config.model_num
    @property
    def model_test_dates(self) -> np.ndarray: return self.trainer.data.model_test_dates
    @property
    def speedup2x(self) -> bool:
        try:
            return self.config.callback_kwargs['ResetOptimizer']['speedup2x']
        except KeyError:
            return False
    @property
    def last_lr(self) -> float |Any:
        if hasattr(self.trainer.model , 'optimizer'):
            return self.trainer.model.optimizer.last_lr
        return 0.
    def display(self , *args , **kwargs):
        if (self.show_info_step or self.initial_models):
            return Logger.stdout(*args , **kwargs)
        else:
            return Logger.log_only(*args , **kwargs)
    def event_sdout(self , event) -> str:
        if event == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.status.epoch} , effective at epoch {self.status.epoch + 1}'
            if self.speedup2x: 
                sdout += ', and will speedup2x'
        elif event == 'new_attempt':
            sdout = '{attempt} {epoch} : {status}, Next attempt goes!'.format(**self.record_texts)
        elif event == 'nanloss':
            sdout = 'Model {model_date}.{model_num} Attempt{attempt}, epoch{epoch} got nanloss!'.format(**self.status.as_dict())
        else:
            raise KeyError(event)
        return sdout

    def on_fit_model_start(self):
        self.model_start_time = datetime.now()
        self.record_model_stage += 1
        self.record_epoch_model = 0
        self.record_texts = {k:'' for k in ['model','attempt','epoch','epoch_model','status','time','exit']}
        self.record_texts['model'] = '{:s} #{:d} @{:4d}'.format(self.config.model_name , self.status.model_num , self.status.model_date)
    
    def on_fit_epoch_start(self):
        self.record_epoch_model += 1
        self.record_epoch_stage += 1
        self.record_texts['epoch'] = 'Ep#{:3d}'.format(self.status.epoch)
        self.record_texts['epoch_model'] = 'Ep#{:3d}'.format(self.record_epoch_model)
        self.record_texts['attempt'] = f'FirstBite' if self.status.attempt == 0 else f'Retrain#{self.status.attempt}'
    
    def on_validation_epoch_end(self):
        if self.trainer.status.epoch < 0: 
            return
        self.record_texts['status'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, best{: .4f}, lr{:.1e}'.format(
            self.metrics.attempt_metrics.latest('train' , 'loss') , 
            self.metrics.attempt_metrics.latest('train' , 'accuracy') ,
            self.metrics.attempt_metrics.latest('valid' , 'accuracy') , 
            self.metrics.best_epoch_metric , 
            self.last_lr)
        if self.status.epoch % self.show_info_step == 0: 
            self.display('{attempt} {epoch} : {status}'.format(**self.record_texts))
    
    def on_fit_epoch_end(self):
        if self.status.fit_loop_breaker: 
            self.record_texts['exit'] = self.status.fit_loop_breaker.trigger_reason
        while self.status.epoch_event: 
            self.display(self.event_sdout(self.status.epoch_event.pop()))
    
    def on_fit_model_end(self):
        model_time_cost = (datetime.now() - self.model_start_time).total_seconds()
        train_accu = self.metrics.attempt_metrics.latest('train' , 'accuracy')
        valid_accu = self.metrics.attempt_metrics.latest('valid' , 'accuracy')
        if self.status.attempt > 0:
            best_accu = f'{self.status.best_attempt[1]: .4f} at attempt#{self.status.best_attempt[0]}'
        else:
            best_accu = f'{self.status.best_attempt[1]: .4f}'
        self.record_texts['status'] = f'Train{train_accu: .4f} Valid{valid_accu: .4f} BestVal{best_accu}'
        self.record_texts['time'] = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format(
            model_time_cost / 60 , model_time_cost / (self.record_epoch_model + 1))
        model_info = '{model}|{attempt} {epoch_model} {exit}|{status}|{time}'.format(**self.record_texts)
        Logger.remark(model_info)
        self.trainer.writer.add_text('Model Info' , model_info)

    def on_after_fit_end(self):  
        fit_time_cost = (datetime.now() - self.status.start_times['fit']).total_seconds()
        if self.status.stage == 'fit' and self.record_model_stage * self.record_epoch_stage:
            per_model = fit_time_cost / 60 / self.record_model_stage
            per_epoch = fit_time_cost / self.record_epoch_stage
            time_cost_info = f'Cost {fit_time_cost / 3600:.1f} Hours, {per_model:.1f} Min/model, {per_epoch:.1f} Sec/Epoch'
            Logger.highlight(f'Finish Process [Fit], {time_cost_info}')
    
    def on_before_test_end(self): 
        time_cost = datetime.now() - self.status.start_times['test']
        Logger.note(f'In Stage [{self.status.stage}], Finish iterating test batches! Cost {Duration(time_cost)}' , vb_level = 3)

    def on_train_batch_end(self):  
        if self.dataloader_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.batch_loss:.5f}')
            # self.device.status()
    def on_train_batch_start(self):
        if self.dataloader_info: 
            self.dataloader.display(f'Train Ep#{self.status.epoch:3d} loss : {self.metrics.batch_loss:.5f}')

    def on_validation_batch_end(self):   
        if self.dataloader_info: 
            self.dataloader.display(f'Valid Ep#{self.status.epoch:3d} accuracy : {self.metrics.batch_accuracy:.5f}')

    def on_test_batch_end(self):         
        if self.dataloader_info: 
            self.dataloader.display('Test {} {} accuracy : {:.5f}'.format(
                self.status.model_submodel , self.trainer.batch_dates[self.batch_idx] , self.metrics.batch_accuracy))

class SummaryWriter(BaseCallBack):
    """record metrics and parameters to tensorboard , and export test summary to a log file"""
    DEBUG_STEP : int = 100 # record debug mode every step in batches
    HIDDEN_FEATURE_MODE : bool = True # enable hidden feature mode to record hidden features stats (accuracy and correlation)
    HIDDEN_FEATURE_STEP : int = 5 # record hidden features every step in epoches
    SUMMARY_NDIGITS : ClassVar[dict[str,int]] = {'Avg':4,'Sum':2,'Std':4,'T':2,'IR':4}

    def __init__(self , trainer , **kwargs) -> None:
        super().__init__(trainer , **kwargs)
        self.init_time = datetime.now()
        if self.HIDDEN_FEATURE_MODE:
            self.hidden_accuracies : list[torch.Tensor] = []
            self.hidden_correlations : list[torch.Tensor] = []
        
    def __repr__(self):
        return f'{self.__class__.__name__}(trainer={self.trainer})'

    @property
    def summary_log_file(self):
        log_name = 'st_results' if self.config.base_path.is_short_test else 'results'
        return LogFile.initialize('model' , 'summary' , log_name)

    @property
    def last_lr(self) -> float |Any:
        if hasattr(self.trainer.model , 'optimizer'):
            return self.trainer.model.optimizer.last_lr
        return 0.

    @property
    def epoch(self) -> int:
        return self.status.epoch_model

    @property
    def batch_idx(self) -> int:
        return self.trainer.batch_idx

    @property
    def writer(self):
        return self.trainer.writer

    def named_parameters(self) -> list[tuple[str,nn.Parameter]]:
        net = self.trainer.model.net
        if not isinstance(net , nn.Module):
            return []
        return [(name , param) for name , param in net.named_parameters()]

    def step_index(self , step_type : str = 'batch'):
        if step_type == 'epoch':
            return self.epoch
        elif step_type == 'batch':
            return self.epoch * self.trainer.batch_num + self.batch_idx
        else:
            raise ValueError(f'Invalid step type: {step_type}')

    def add_metrics(self , step_type : str = 'epoch'):
        step = self.step_index(step_type)
        accuracies , losses = self.metrics.epoch_accuracies , self.metrics.epoch_losses
        if not isinstance(accuracies , dict):
            accuracies = {'Accuracy':accuracies}
        if not isinstance(losses , dict):
            losses = {'Loss':losses}
        self.writer.add_scalars(f'00.Metrics/Accuracy/{self.status.dataset.title()}' , accuracies , step)
        self.writer.add_scalars(f'00.Metrics/Loss/{self.status.dataset.title()}' , losses , step)

    def add_metric(self , metric_type: Literal['accuracy','loss'] , metric : Any | dict[str,Any] , step_type : str = 'epoch'):
        main_tag = f'00.Metrics/{metric_type.title()}/{self.status.dataset.title()}'
        values = metric if isinstance(metric , dict) else {metric_type:metric}
        self.writer.add_scalars(main_tag , values , self.step_index(step_type))

    def add_hidden_accuracy(self , hidden_accuracies : torch.Tensor , step_type : str = 'epoch'):
        step = self.step_index(step_type)
        self.writer.add_histogram(f'01.HiddenFeatures/Accuracy/{self.status.dataset.title()}' , hidden_accuracies , step)

    def add_hidden_correlation(self , hidden_correlation : torch.Tensor , step_type : str = 'epoch'):
        step = self.step_index(step_type)
        corr_heatmap = self.hidden_correlation_to_chw(hidden_correlation)
        self.writer.add_image(f'01.HiddenFeatures/Correlation/{self.status.dataset.title()}' , corr_heatmap , step , dataformats='CHW')

    def add_lr(self , step_type : str = 'epoch'):
        assert self.status.dataset == 'train' , 'lr is only supported for train dataset'
        self.writer.add_scalar('02.HyperParameter/LearnRate' , self.last_lr , self.step_index(step_type))
        
    def add_weight_norm(self , step_type : str = 'epoch'):
        assert Proj.debug_mode and self.status.dataset == 'train' , f'DEBUG_MODE is not enabled or dataset is not train : {Proj.debug_mode} {self.status.dataset}'
        step = self.step_index(step_type)
        [self.writer.add_scalar(f'03.ModelWeights/Norm/{name}' , torch.norm(param.data , p=2) , step) for name , param in self.named_parameters() if param.data is not None]

    def add_weight_histogram(self , step_type : str = 'batch'):
        assert Proj.debug_mode and self.status.dataset == 'train' , f'DEBUG_MODE is not enabled or dataset is not train : {Proj.debug_mode} {self.status.dataset}'
        step = self.step_index(step_type)
        [self.writer.add_histogram(f'03.ModelWeights/Histogram/{name}' , param.data , step) for name , param in self.named_parameters() if param.data is not None]

    def add_grad_norm(self , step_type : str = 'epoch'):
        assert Proj.debug_mode and self.status.dataset == 'train' , f'DEBUG_MODE is not enabled or dataset is not train : {Proj.debug_mode} {self.status.dataset}'
        step = self.step_index(step_type)
        [self.writer.add_scalar(f'04.ModelGradients/Norm/{name}' , torch.norm(param.grad , p=2) , step) for name , param in self.named_parameters() if param.grad is not None]

    def add_grad_histogram(self , step_type : str = 'batch'):
        assert Proj.debug_mode and self.status.dataset == 'train' , f'DEBUG_MODE is not enabled or dataset is not train : {Proj.debug_mode} {self.status.dataset}'
        step = self.step_index(step_type)
        [self.writer.add_histogram(f'04.ModelGradients/Histogram/{name}' , param.grad , step) for name , param in self.named_parameters() if param.grad is not None]

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
        with tarfile.open(tar_filename, 'w:gz') as tar:
            for path in ts_folder.iterdir():
                tar.add(path, arcname=path.relative_to(ts_folder))

    @classmethod
    def format_messages(cls , messages : dict[str, str | dict] , indent : int = 0) -> list[str]:
        key_len = max(len(key) for key in messages.keys())
        msgs : list[str] = []
        for key , msg in messages.items():
            if isinstance(msg , dict):
                msgs.append('  ' * indent + f'{key:{key_len}s} : ')
                msgs.extend(cls.format_messages(msg , indent + 1))
            else:
                msgs.append('  ' * indent + f'{key:{key_len}s} : {msg}')
        return msgs
    
    @staticmethod
    def hidden_correlation_to_chw(hcorr : torch.Tensor) -> torch.Tensor:
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

    def append_batch_hidden_summary(self):
        if self.HIDDEN_FEATURE_MODE and 'hidden' in self.batch_data.output.other:
            self.hidden_accuracies.append(self.metrics.accuracy_function.hidden_accuracies(self.batch_data))
            self.hidden_correlations.append(torch.corrcoef(self.batch_data.output.hidden.T))

    def add_hidden_summaries(self , step_type : str = 'epoch'):
        if self.HIDDEN_FEATURE_MODE:
            if self.hidden_accuracies:
                hidden_accuracies = torch.stack(self.hidden_accuracies , -1).mean(dim=-1)
                self.hidden_accuracies.clear()
                self.add_hidden_accuracy(hidden_accuracies , step_type)

            if self.hidden_correlations:
                hidden_correlations = torch.stack(self.hidden_correlations , -1).mean(dim=-1)
                self.hidden_correlations.clear()
                self.add_hidden_correlation(hidden_correlations , step_type)

    def on_before_clip_gradients(self):
        if Proj.debug_mode and self.batch_idx % self.DEBUG_STEP == 0:
            self.add_weight_histogram('batch')
            self.add_grad_histogram('batch')

    def on_train_epoch_start(self):
        self.add_lr('epoch')

    def on_train_batch_end(self):
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.append_batch_hidden_summary()

    def on_train_epoch_end(self):
        self.add_metrics('epoch')
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.add_hidden_summaries('epoch')

    def on_validation_batch_end(self):
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.append_batch_hidden_summary()

    def on_validation_epoch_end(self):
        self.add_metrics('epoch')
        if self.HIDDEN_FEATURE_MODE and self.epoch % self.HIDDEN_FEATURE_STEP == 0:
            self.add_hidden_summaries('epoch')

    def on_after_fit_end(self):
        self.pack_tensorboard_dir()

    def on_summarize_model(self):
        """pack tensorboard dir and export test summary to json"""
        if self.status.test_summary.empty: 
            return
        
        test_name = f'{self.config.base_path.full_name}'

        duration : dict[str,str] = {
            stage : f'{Duration(self.status.end_times[stage] - self.status.start_times[stage])}' for stage in self.config.stage_queue
        }
        metrics : dict[str,str] = {}
        for col in self.status.test_summary.columns:
            series = self.status.test_summary[col]
            key = '{}.{}'.format(*col)
            value = '|'.join([f'{k}({series.round(v).loc[k]})' for k,v in self.SUMMARY_NDIGITS.items() if k in series.index])
            metrics[key] = value

        messages = {
            f'model' : f'{self.config.model_name} x {len(self.config.model_num_list)})',
            f'submodel' : f'{self.config.submodels}',
            f'start' : f'{self.init_time.strftime("%Y-%m-%d %H:%M:%S")}',
            f'stages' : f'{self.config.stage_queue}',
            f'inputs' : self.data.input_keys_subkeys,
            f'labels' : f'{self.config.labels}',
            f'range' : f'{self.config.beg_date} - {self.config.end_date}',
            f'duration' : duration,
            f'metrics' : metrics,
        }
        msgs = self.format_messages(messages , indent = 0)
        self.summary_log_file.write(test_name , *msgs)
    