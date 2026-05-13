from __future__ import annotations

from datetime import datetime
from .base_trainer import BaseTrainer , ModelStreamLineWithTrainer

class TrainerTexts(ModelStreamLineWithTrainer):
    """Status texts class, used to generate the texts of the status"""
    def __init__(self , trainer : BaseTrainer):
        self.bound_with_trainer(trainer)
    @property
    def model(self) -> str:
        return f'{self.config.model_name} #{self.model_num:d} @{self.model_date:4d}'
    @property
    def attempt(self) -> str:
        return f'{self.status.attempt_key}'
    @property
    def model_epoch(self) -> str:
        return f'ModelEpoch#{self.status.model_epoch}'
    @property
    def exit(self) -> str:
        if self.status.loop_end:
            return self.status.end_attempt_event.reason
        else:
            return ''
    @property
    def info(self) -> str:
        last_lr = self.trainer.model.optimizer.last_lr if hasattr(self.trainer.model , 'optimizer') else 0.
        best_epoch = self.metrics.attempt_metrics.best_epoch()
        if best_epoch is None:
            return 'Loss{:.4f}, TrainIC{:.4f}, ValidIC{:.4f}, LR{:.1e}'
        valid_accuracies = f'{{{",".join(f"{k}:{v:.4f}" for k,v in best_epoch.valid_accuracies.items())}}}'
        return 'Loss={:.4f}, TrainIC={:.4f}, ValidIC={:.4f}, Best at {:s} with Accu={:s} IC={:.4f}, LR={:.1e}'.format(
            self.metrics.attempt_metrics.latest('train' , 'loss') , 
            self.metrics.attempt_metrics.latest('train' , 'rankic') ,
            self.metrics.attempt_metrics.latest('valid' , 'rankic') , 
            best_epoch.epoch_key , valid_accuracies , best_epoch.valid_rankic , last_lr)
    @property
    def progress(self) -> str:
        return f'{self.attempt} {self.status.epoch_key}: {self.info}'

    @property
    def model_summary(self) -> str:
        model_time_cost = (datetime.now() - self.status.times['model_start']).total_seconds()
        per_epoch = model_time_cost / (self.status.model_epoch + 1)
        best_attempt = self.metrics.model_metrics.best_attempt()
        best_metrics = self.metrics.model_metrics.best_attempt_metrics()
        final_metrics = f'BestAttempt {best_attempt} TrainIC{best_metrics.train_rankic: .4f} ValidIC{best_metrics.valid_rankic: .4f}'
        final_time = f'Cost{model_time_cost / 60:5.1f}Min,{per_epoch:5.1f}Sec/Ep'
        return f'{self.model}|{self.attempt} {self.model_epoch} {self.exit}|{final_metrics}|{final_time}'
    @property
    def fit_summary(self) -> str:
        if self.is_fitting and self.status.total_epochs * self.status.total_models:
            fit_time_cost = (datetime.now() - self.status.times['fit_start']).total_seconds()
            per_model = fit_time_cost / 60 / self.status.total_models
            per_epoch = fit_time_cost / self.status.total_epochs
            return f'Cost {fit_time_cost / 3600:.1f} Hours, {per_model:.1f} Min/model, {per_epoch:.1f} Sec/Epoch'
        else:
            return ''