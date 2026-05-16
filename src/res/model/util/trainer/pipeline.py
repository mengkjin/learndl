"""
This module is used to define the base class for all model pipelines , e.g. trainer, predictor, data module, etc.
A typical pipeline is a sequence of hooks that are executed in order of a fitting and testing process.

trainer.__init__(): 
only initialize the core components: config , data , model , callbacks

[Stage Setup]
|- hook: on_configure_model
|- init_utils: status , record , texts , container , metrics , checkpoint , deposition , writer

[Stage Data]
|- hook: on_data_start_before
|- hook: on_data_start
|- hook: on_data_start_after
|- load_data: data.load_data()
|- hook: on_before_data_end
|- hook: on_data_end
|- hook: on_data_end_after

[Stage Fit]
|- hook: on_fit_start_before
|- hook: on_fit_start
|- hook: on_fit_start_after
|- for model_date , model_num in iter_model_num_date():
|--- hook: on_fit_model_date_start if model_num == 0
|--- hook: on_fit_model_start
|--- for epoch in iter_fit_epoches():
|----- hook: on_fit_epoch_start
|----- hook: on_train_epoch_start
|----- for batch in iter_train_dataloader():
|------- hook: on_train_batch_start
|------- batch_forward()
|------- hook: on_train_batch_end
|------- hook: on_batch_metrics_before
|------- batch_metrics()
|------- hook: on_batch_metrics_after
|------- hook: on_batch_backward_before
|------- batch_backward()
|------- hook: on_before_clip_gradients
|------- clip_gradients()
|------- hook: on_batch_backward_after
|----- hook: on_fit_epoch_end_before
|----- hook: on_train_epoch_end
|----- hook: on_validation_epoch_start
|----- for batch in iter_val_dataloader():
|------- hook: on_validation_batch_start
|------- batch_forward()
|------- hook: on_batch_metrics_before
|------- batch_metrics()
|------- hook: on_batch_metrics_after
|------- hook: on_validation_batch_end
|----- hook: on_validation_epoch_end
|----- hook: on_fit_epoch_end_before
|----- hook: on_fit_epoch_end
|--- hook: on_before_save_model
|--- hook: on_fit_model_end
|--- hook: on_fit_model_date_end if model_num == model_num
|- hook: on_fit_end_before
|- hook: on_fit_end
|- hook: on_fit_end_after
|- special hook: on_new_attempt
|- special hook: on_new_phase

[Stage Test]
|- hook: on_test_start_before
|- hook: on_test_start
|- hook: on_test_start_after
|- for model_date , model_num in iter_model_num_date():
|--- hook: on_test_model_date_start if model_num == 0
|--- hook: on_test_model_start
|--- for submodel in iter_model_submodels():
|----- hook: on_test_submodel_start
|----- for batch in iter_test_dataloader():
|------- hook: on_test_batch_start
|------- batch_forward()
|------- hook: on_test_batch_end
|----- hook: on_test_submodel_end
|--- hook: on_test_model_end
|--- hook: on_test_model_date_end if model_num == model_num
|- hook: on_test_end_before
|- hook: on_test_end
|- hook: on_test_end_after

[Summarizing]
|- on_summarize_model
"""

from __future__ import annotations

from abc import ABC , ABCMeta

from src.proj import Logger
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import BatchOutput

__all__ = ['BasePipeline' , 'TrainerPipeline']

class _Pipeline(ABC):
    """Base class for all model pipelines , e.g. trainer, predictor, data module, etc."""
    @staticmethod
    def base_hooks():
        return [hook for hook in dir(_Pipeline) if hook.startswith('on_')]

    # [Stage Setup]
    def on_configure_model(self): ...
    
    # [Stage Data]
    def on_data_start_before(self): ...
    def on_data_start(self): ... 
    def on_data_start_after(self): ...
    def on_data_end_before(self): ...
    def on_data_end(self): ... 
    def on_data_end_after(self): ...
    
    # [Stage Fit]
    def on_fit_start_before(self): ...
    def on_fit_start(self): ...
    def on_fit_start_after(self): ...
    def on_fit_end_before(self): ...
    def on_fit_end(self): ...
    def on_fit_end_after(self): ...
    # [~ Model Iteration]
    def on_fit_model_date_start(self): ...
    def on_fit_model_start(self): ...
    def on_before_save_model(self): ...
    def on_fit_model_end(self): ...
    def on_fit_model_date_end(self): ...
    # [~ Epoch Iteration]
    def on_fit_epoch_start(self): ...
    def on_train_epoch_start(self): ...
    def on_train_epoch_end(self): ...
    def on_validation_epoch_start(self): ...
    def on_validation_epoch_end(self): ...
    def on_fit_epoch_end_before(self): ... 
    def on_fit_epoch_end(self): ...
    # [~ Batch Iteration]
    def on_train_batch_start(self): ...
    def on_train_batch_end(self): ...
    def on_validation_batch_end(self): ... 
    def on_validation_batch_start(self): ...
    def on_batch_metrics_before(self): ... 
    def on_batch_metrics_after(self): ... 
    def on_batch_backward_before(self): ...
    def on_before_clip_gradients(self): ...
    def on_batch_backward_after(self): ...
    # [~ Special Hooks]
    def on_new_attempt(self): ...
    def on_new_phase(self): ...

    # [Stage Test]
    def on_test_start_before(self): ...
    def on_test_start(self): ...
    def on_test_start_after(self): ...
    def on_test_end_before(self): ...
    def on_test_end(self): ...
    def on_test_end_after(self): ...
    # [~ Model Iteration]
    def on_test_model_date_start(self): ...
    def on_test_model_start(self): ...
    def on_test_model_end(self): ...
    def on_test_model_date_end(self): ... 
    # [~ Submodel Iteration]
    def on_test_submodel_end(self): ... 
    def on_test_submodel_start(self): ... 
    # [~ Batch Iteration]
    def on_test_batch_end(self): ...
    def on_test_batch_start(self): ... 

    # [Summarizing]
    def on_summarize_model(self): ...

class _PipelineMeta(ABCMeta): 
    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)

        # check if the class defines extra hooks
        current = set([hook for hook in dir(new_cls) if hook.startswith('on_')])
        base_hooks = set(getattr(new_cls, 'base_hooks', lambda: [])())
        extra = current - base_hooks
        if extra:
            Logger.warning(f'{new_cls.__name__} defines extra hooks: {extra}')

        # check if the class defines placeholdernames for hooks
        placeholder_names = set([hook for hook in dir(new_cls) if hook.startswith(('_raw_'))])
        if placeholder_names:
            Logger.warning(f'{new_cls.__name__} defines placeholder names for hooks: {placeholder_names}')
        return new_cls

class BasePipeline(_Pipeline, metaclass=_PipelineMeta):
    """Base class for all model pipelines , e.g. trainer, predictor, data module, etc."""
    def execute_hook(self , hook : str):
        getattr(self , hook)()

class TrainerPipeline(BasePipeline):
    def reset(self):
        self._trainer = None
        self._config = None
        return self

    def bound_with(self , binder):
        if isinstance(binder , ModelConfig):
            return self.bound_with_config(binder)
        else:
            return self.bound_with_trainer(binder)

    def bound_with_config(self , config : ModelConfig):
        assert not getattr(self , '_trainer' , None) , 'Cannot bound with config if bound with trainer first'
        self._config = config
        return self

    def bound_with_trainer(self , trainer):
        self.reset()
        from src.res.model.util.trainer import BaseTrainer
        assert isinstance(trainer , BaseTrainer) , f'trainer must be an instance of BaseTrainer, but got {type(trainer)}'
        self._trainer = trainer
        return self

    @property
    def trainer(self):
        from src.res.model.util.trainer import BaseTrainer
        if not hasattr(self , '_trainer') or self._trainer is None:
            raise ValueError('Trainer is not bound')
        else:
            assert isinstance(self._trainer , BaseTrainer) , f'trainer must be an instance of BaseTrainer, but got {type(self._trainer)}'
        return self._trainer

    @property
    def config(self) -> ModelConfig:
        config = self.trainer.config if getattr(self , '_trainer' , None) else self._config
        assert isinstance(config , ModelConfig) , f'config must be an instance of ModelConfig, but got {type(config)}'
        return config
    @property
    def model(self): 
        return self.trainer.model
    @property
    def data(self): 
        return self.trainer.data
    @property
    def status(self):  
        return self.trainer.status
    @property
    def callback(self): 
        return self.trainer.callback
    @property
    def container(self): 
        return self.trainer.container
    @property
    def metrics(self):  
        return self.trainer.metrics
    @property
    def checkpoint(self): 
        return self.trainer.checkpoint
    @property
    def deposition(self): 
        return self.trainer.deposition
    @property
    def texts(self): 
        return self.trainer.texts
    @property
    def record(self): 
        return self.trainer.record
    @property
    def device(self): 
        return self.trainer.device
    @property
    def base_path(self): 
        return self.trainer.base_path
    @property
    def batch_input(self): 
        return self.trainer.batch_input
    @property
    def batch_idx(self): 
        return self.trainer.batch_idx
    @property
    def batch_output(self): 
        return self.trainer.batch_output
    @batch_output.setter
    def batch_output(self , value : BatchOutput): 
        self.trainer.batch_output = value
    @property
    def batch_data(self): 
        return self.trainer.batch_data
    @property
    def model_date(self): 
        return self.trainer.model_date
    @property
    def model_num(self): 
        return self.trainer.model_num
    @property
    def model_submodel(self): 
        return self.trainer.model_submodel
    @property
    def model_str(self): 
        return self.trainer.model_str
    @property
    def is_fitting(self): 
        return self.trainer.is_fitting
    @property
    def batch_key(self): 
        return self.batch_idx if self.is_fitting else self.trainer.batch_dates[self.batch_idx]
