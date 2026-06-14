"""
ModelConfg: The all-in-one config class for model training and inference.
It includes:
1. model config
2. algo config
3. schedule config
"""

from __future__ import annotations
import os, random, torch
import numpy as np

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Type, cast

from src.proj import PATH, MACHINE, Const, Proj , Base
from src.proj.util.functional.device import Device
from src.res.algo import AlgoModule
from src.res.factor.calculator import StockFactorHierarchy, FactorCalculator

from src.res.model.util.core import ModelPath , model_module_type, is_null_module_type

__all__ = ['ModelConfig']

def get_config_dict(input: dict | Path | list[Path] | Base.FlattenDict | None) -> Base.FlattenDict:
    """
    get flattened config dict from input
    keep nested keys if they are in the input or are not all lowercase
    """
    def keep_nested(k: str) -> bool:
        exclude_keys = ('input.sequence.lens' , 'input.sequence.steps' , 'input.factor.types' ,
                        'train.criterion.loss' , 'train.criterion.accuracy' , 'train.criterion.multilosses')
        return (k.endswith(exclude_keys) or not k.islower())
    if isinstance(input, Base.FlattenDict):
        return input
    else:
        return Base.FlattenDict.from_input(input , keep_nested = keep_nested)

class ScheduleConfig(Base.BoundLogger , Base.CacheProps):
    """load schedule config from config/model/schedule or .local_resources/shared/schedule_model/schedule or the model's base_path"""
    def __init__(self, base_path: ModelPath | None = None, schedule_name: str | None = None, model_name: Any | None = None , * , indent: int = 1 , vb_level: Any = 2, **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.base_path = base_path
        self.schedule_name = schedule_name
        if model_name:
            assert model_name == schedule_name or not ScheduleConfig.check_name_exist(model_name), \
                f"model_name [{model_name}] is not the same as schedule_name [{schedule_name}]"
        self.Param = self.get_config_dict(self.base_path, self.schedule_name)

    def __getitem__(self, key: str) -> Any:
        return self.Param[key]

    def __setitem__(self, key: str, value: Any):
        self.Param[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.Param.get(key, default)

    def get_config_dict(self , base_path: ModelPath | None, schedule_name: str | None) -> Base.FlattenDict:
        config_path = self.find_path(base_path, schedule_name)
        config = get_config_dict(config_path)
        if not base_path and config:
            self.logger.alert1(f'Using schedule name "{schedule_name}" to load config' , vb = 1)
        if schedule_name and config:
            if 'model.name' in config and config['model.name'] != schedule_name:
                self.logger.alert2(f'model.name {config['model.name']} is not the same as schedule_name {schedule_name}' , vb = 1)
            config.update({'model.name': schedule_name} , relevant_only = False)
        return config

    @classmethod
    def find_path(cls , base_path: ModelPath | None, name: str | None) -> Path | None:
        if base_path:
            config_path = base_path.conf_file("schedule")
            return config_path if config_path.exists() else None
        elif not name:
            return None
        else:
            schedule_path_0 = PATH.sched.joinpath(f"{name}.yaml")
            schedule_path_1 = PATH.sched_shared.joinpath(f"{name}.yaml")
            assert not (schedule_path_0.exists() and schedule_path_1.exists()), \
                f"{name} exists in both config/model/schedule and .local_resources/shared/schedule_model/schedule"
            if schedule_path_0.exists():
                return schedule_path_0
            elif schedule_path_1.exists():
                return schedule_path_1
            else:
                return None

    @classmethod
    def check_name_exist(cls , model_name: str | None = None) -> bool:
        if not model_name:
            return False
        path = cls.find_path(None, model_name)
        return True if path else False

class BaseModelConfig(Base.BoundLogger , Base.CacheProps):
    CONFIG_LIST = ["env", "model", "input", "train", "callbacks", "conditional"]
    REQUIRED_CONFIG_PARAM = get_config_dict(PATH.conf.joinpath("model", "default", "required.yaml"))
    OPTIONAL_CONFIG_PARAM = get_config_dict(PATH.conf.joinpath("model", "default", "optional.yaml"))

    def __init__(
        self, base_path: ModelPath | Base.strPath | None, *,
        module: str | None = None, schedule_name: str | None = None, override=None, 
        indent: int = 1 , vb_level: Any = 2, **kwargs,
    ):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.base_path = ModelPath(base_path)
        self.start_with_none = not self.base_path
        self.force_module = module
        self.schedule_name = schedule_name
        self.override = (override or {}) | kwargs
        self.load_params()
        self.override_params()
        self.check_validity()
        
    def __bool__(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(base_path={self.base_path})"

    def __getitem__(self, key: str) -> Any:
        if key in self.REQUIRED_CONFIG_PARAM:
            return self.Param[key]
        else:
            return self.Param.get(key, self.OPTIONAL_CONFIG_PARAM.get(key))

    def __setitem__(self, key: str, value: Any):
        self.Param[key] = value

    def resumed_config_param(self) -> Base.FlattenDict | None:
        if (self.base_path and not self.base_path.is_null_model and not self.short_test):
            conf_file = self.base_path.conf_file("model")
            return get_config_dict(conf_file) if conf_file.exists() else None
        else:
            return None

    def default_config_param(self) -> Base.FlattenDict:
        return self.REQUIRED_CONFIG_PARAM.combine_with(self.OPTIONAL_CONFIG_PARAM)
        
    def current_config_param(self) -> Base.FlattenDict:
        return get_config_dict([PATH.conf.joinpath("model", f"{cfg}.yaml") for cfg in self.CONFIG_LIST])

    def optional_load_params(self , option : Literal["current", "default"]):
        Param = None if self.start_with_none else self.resumed_config_param()
        if Param is None:
            Param = self.current_config_param() if option == "current" else self.default_config_param()
        return Param

    def load_params(self):
        if self.force_module:
            assert not self.schedule_name, \
                f"force_module [{self.force_module}] is provided, but schedule_name [{self.schedule_name}] is also provided"
            assert not self.base_path.full_module_name, \
                f"force_module [{self.force_module}] is provided, but base_path.full_module_name [{self.base_path.full_module_name}] is also provided"

        if self.schedule_name:
            # case 1: with schedule name, load schedule config first, then resume or load default config
            assert ScheduleConfig.check_name_exist(self.schedule_name), f"schedule_name [{self.schedule_name}] does not exist"
            self.Param = self.optional_load_params("default")
            self['model.name'] = self.schedule_name
        else:
            # case 2: without schedule name, resume or load current config first, then adjust according to force_module, then check schedule name conflict
            self.Param = self.optional_load_params("current")
            if self.force_module:
                self.logger.alert1(f"force_module [{self.force_module}] is provided, will use it to load config" , vb = 1)
                self['model.module'] = self.force_module
                self['model.name'] = ''
            assert self.base_path or not ScheduleConfig.check_name_exist(self['model.name']), \
                f"base_path is not provided, but model.name [{self['model.name']}] is owned by a schedule model, and must not be used as a model name"
        self.schedule_config = ScheduleConfig(self.base_path, self.schedule_name, indent=self.indent, vb_level=self.vb_level)
        return self

    def override_params(self):
        model_module_candidate = {
            "base_path": self.base_path.full_module_name,
            "force_module": str(self.force_module).lower().replace(" ", "").replace("/", "@") if self.force_module else None,
            "schedule": self.schedule_name and self.schedule_config.get("model.module", None),
        }
        assert len(np.unique([v for v in model_module_candidate.values() if v])) <= 1, (
            f"only one of base_path , force_module , schedule can be provided, but got {model_module_candidate}"
        )
        model_modules = [v for v in model_module_candidate.values() if v]
        if model_modules:
            self["model.module"] = model_modules[0]

        # deal with short_test given short_test model path / override / should be short_test
        if self.base_path and self.base_path.is_short_test:
            self.override["env.short_test"] = True
        elif (short_test := self.override.pop("short_test", None)) is not None:
            self.override["env.short_test"] = short_test
        elif self.should_be_short_test and ("env.short_test" not in self.override):
            self.override["env.short_test"] = True
        if "env.short_test" in self.override:
            self.Param['env.short_test'] = self.override.pop("env.short_test")
        if self.short_test:
            self.logger.alert1(f'Short test is enabled, will update conditional config' , vb = 1)
            self.Param.update(self.Param.get("conditional.short_test", {}))

        self.Param.update(self.schedule_config.Param)
        if self.model_module == "transformer":
            self.logger.alert1(f'Model module is transformer, will update conditional config' , vb = 1)
            self.Param.update(self.Param.get("conditional.transformer", {}))
        
        self.Param.update(self.override)
        return self

    def check_validity(self):
        if not self.base_path:
            full_name = self.generate_model_full_name()
            self.base_path.with_full_name(full_name)
        else:
            full_name = self.base_path.full_name

        # check base_path is set correctly
        assert self.base_path, f"base_path is still not set after generating model full name {full_name}"
        assert self.base_path.module_type == self.module_type, f"module_type {self.module_type} is not the same as base_path.module_type {self.base_path.module_type}"
        assert self.base_path.model_module == self.model_module, f"model_module {self.model_module} is not the same as base_path.model_module {self.base_path.model_module}"

        # check short_test is set correctly
        if self.should_be_short_test and not self.short_test:
            self.logger.alert1("Should be at server or short_test, but short_test is False now!" , vb = 1)

        # check sample_method is set correctly
        nn_category = AlgoModule.nn_category(self.model_module)
        if nn_category == "tra":
            assert self.sample_method != "total_shuffle", self.sample_method
        if nn_category == "vae":
            assert self.sample_method == "sequential", self.sample_method

        # check nn_datatype is set correctly
        nn_datatype = AlgoModule.nn_datatype(self.model_module)
        if nn_datatype:
            self.input_data_types = nn_datatype

        # check submodels is set correctly
        if self.module_type != "nn" or self.boost_head:
            self["model.submodels"] = ["best"]
        if "best" not in self.submodels:
            self.submodels.insert(0, "best")

        # check input_type is set correctly
        if self.input_type != "data" or self.module_type != "nn":
            assert self.sample_method == "sequential", self.sample_method
        if self.module_type == "factor":
            self["input.type"] = "factor"
            self["input.factor.types"] = []
            self["input.sequence.lens"] = self["input.sequence.lens"] | {"factor": 1}

        # check missing required keys
        missing_required_keys = np.setdiff1d(list(self.REQUIRED_CONFIG_PARAM.keys()), list(self.Param.keys())).tolist()
        if missing_required_keys:
            self.logger.error(f"{missing_required_keys} are required but not in config files")
            raise ValueError(f"{missing_required_keys} are required but not in config files")

        redundant_keys = np.setdiff1d(list(self.Param.keys()), list(self.REQUIRED_CONFIG_PARAM.keys()) + list(self.OPTIONAL_CONFIG_PARAM.keys())).tolist()
        if redundant_keys:
            self.logger.alert1(f"{redundant_keys} in config files are not in default config params" , vb = 1)

        return self

    def generate_model_full_name(self):
        if is_null_module_type(self.module_type):
            full_name = self.full_module_name
        else:
            model_name = str(self["model.name"]) if self["model.name"] else self.generate_valid_model_name()
            full_name = f"{self.full_module_name}@{model_name}"
        if self.short_test:
            full_name = f"st@{full_name}"
        return full_name

    def generate_valid_model_name(self):
        mod_str = self.model_module.removeprefix(f"{self.module_type}@")
        head_str = "boost" if self.boost_head else None
        if self.input_type == "data":
            data_str = "_".join(self.input_data_types)
        else:
            data_str = self.input_type
        model_name = "_".join([s for s in [mod_str, head_str, data_str] if s])
        if ScheduleConfig.check_name_exist(model_name):
            model_name = f'{model_name}.{random.randint(1, 99):02d}'
        return model_name

    def generate_algo_config(self):
        algo_config = AlgoConfig(
            self.base_path,
            start_with_none=self.start_with_none,
            module=self.model_module,
            boost_head=self.boost_head,
            short_test=self.short_test,
            schedule_config=self.schedule_config,
            override={k: v for k, v in self.override.items() if k not in self.Param.keys()},
            indent=self.indent, vb_level=self.vb_level,
        ).expand()
        # reversely update specific params in model_param to self.Param
        self.Param.update(algo_config.Param)
        return algo_config

    @property
    def base_path(self) -> ModelPath:
        return self.cached_properties.get('base_path')

    @base_path.setter
    def base_path(self, value: ModelPath):
        self.cached_properties.set('base_path' , value)

    @property
    def module_type(self):
        return self.full_module_name.split("@")[0]

    @property
    def model_module(self):
        return self.full_module_name.split("@")[1]

    @model_module.setter
    def model_module(self, value: str):
        self["model.module"] = value.lower()
        self.cached_properties.set('full_module_name' , None)

    @property
    def full_module_name(self) -> str:
        """get module_type@model_module out of model configs"""
        if not self.cached_properties.has('full_module_name') or self.cached_properties.get('full_module_name') is None:
            mod_str = str(self["model.module"]).lower().replace(" ", "").replace("/", "@")
            module_type = model_module_type(mod_str)
            if mod_str.startswith(f"{module_type}@"):
                model_module = mod_str.removeprefix(f"{module_type}@")
            elif mod_str == module_type:
                model_module = self[f"model.module.{module_type}"]
                assert model_module, f"model.module.{module_type} is empty!"
            else:
                model_module = mod_str
            assert model_module, f"model_module is empty after parsing for {mod_str}"
            assert "@" not in model_module, f"model_module {model_module} contains @"
            self.cached_properties.set('full_module_name' , f"{module_type}@{model_module}")
        return self.cached_properties.get('full_module_name')

    @property
    def model_name(self) -> str:
        return self.base_path.model_name

    @property
    def model_clean_name(self) -> str:
        return self.base_path.model_clean_name

    @property
    def should_be_short_test(self):
        return not MACHINE.cuda_server

    @property
    def nn_category(self) -> str | None:
        return AlgoModule.nn_category(self.model_module)

    @property
    def special(self) -> dict[str , bool]:
        return self.Param.get("model.special" , {})

    @property
    def short_test(self) -> bool:
        return self.base_path.is_short_test if self.base_path else bool(self["env.short_test"])

    @short_test.setter
    def short_test(self, value: bool):
        self["env.short_test"] = value

    @property
    def random_seed(self) -> int | Any:
        return self["env.random_seed"]

    @property
    def mem_storage(self) -> bool:
        return bool(self["env.mem_storage"])

    @property
    def precision(self) -> Any:
        prec = self["env.precision"]
        return getattr(torch, prec) if isinstance(prec, str) else prec

    @property
    def beg_date(self) -> int:
        return int(self["model.beg_date"])

    @property
    def end_date(self) -> int:
        return int(self["model.end_date"])

    @property
    def labels(self) -> list[str]:
        return self["model.labels"]

    @property
    def submodels(self) -> list[str]:
        return self["model.submodels"]

    @property
    def input_type(self) -> Literal["data", "hidden", "factor", "combo"]:
        assert self["input.type"] in ["data", "hidden", "factor", "combo"], self["input.type"]
        return self["input.type"]

    @property
    def input_filter_secid(self) -> str | None:
        return self["input.filter.secid"]

    @property
    def input_filter_date(self) -> str | None:
        return self["input.filter.date"]

    @property
    def input_data_types(self) -> list[str]:
        if self.input_type == "data" or (self.input_type == "combo" and "data" in self["input.combo.types"]):
            return self.unwrap_inputs(self["input.data.types"])
        else:
            return []

    @input_data_types.setter
    def input_data_types(self, value: list[str] | str):
        self["input.data.types"] = value

    @property
    def input_data_prenorm(self) -> dict[str, Any]:
        return self["input.data.prenorm"]

    @property
    def input_hidden_types(self) -> list[str]:
        if self.input_type == "hidden" or (self.input_type == "combo" and "hidden" in self["input.combo.types"]):
            return self.unwrap_inputs(self["input.hidden.types"])
        else:
            return []

    @property
    def input_factor_types(self) -> list[str]:
        if self.input_type == "factor" or (self.input_type == "combo" and "factor" in self["input.combo.types"]):
            return self.unwrap_inputs(self["input.factor.types"])
        else:
            return []

    @classmethod
    def unwrap_inputs(cls , factors: list | dict | str) -> list[str]:
        if isinstance(factors, str):
            return factors.strip().split("+")
        else:
            if isinstance(factors, dict):
                factors_lists = [cls.unwrap_inputs(f) for f in factors.values()]
            elif isinstance(factors, list):
                factors_lists = [cls.unwrap_inputs(f) for f in factors]
            else:
                raise ValueError(f"Invalid factors type: {type(factors)}")
            return [f for facs in factors_lists for f in facs]

    @property
    def input_factor_names(self) -> list[str]:
        return [self.model_module] if self.module_type == "factor" else []

    @property
    def input_combo_types(self) -> dict[str, list[str]]:
        input_combos = self["input.combo.types"]
        if self.input_type == "combo" and not input_combos:
            raise ValueError("input.combo.types is empty when input_type is combo")
        combos = {k: v for k, v in self.input_keys_all.items() if k in input_combos}
        return combos

    @property
    def input_keys_all(self) -> dict[str, list[str]]:
        return {
            "data": self.input_data_types,
            "factor": self.input_factor_types,
            "hidden": self.input_hidden_types,
        }

    @property
    def input_keys_subkeys(self) -> dict[str, dict[str, str]]:
        return {
            "data": {tp: "." for tp in self.input_data_types},
            "factor": {tp: "." for tp in self.input_factor_types},
            "hidden": {tp: "." for tp in self.input_hidden_types},
        }

    @property
    def window(self) -> int:
        tw = self["model.window"]
        tw = max(int(tw) if tw is not None else 0, 0)
        return tw if tw > 0 else int(self[f"model.window.{self.module_type}"])

    @property
    def interval(self) -> int:
        itv = self["model.interval"]
        itv = max(int(itv) if itv is not None else 0, 0)
        if itv > 0:
            return itv
        else:
            return int(self[f"model.interval.{self.module_type}"])

    @property
    def boost_head(self):
        use_boost_head = bool(self["model.module.nn.boost_head"])
        if use_boost_head:
            return self["model.module.nn.boost_head.boost"]
        else:
            return ""

    @property
    def boost_optuna(self) -> bool:
        return bool(self["model.module.boost.optuna"])

    @property
    def boost_optuna_trials(self) -> int:
        if self["model.module.boost.optuna.trials"]:
            return int(self["model.module.boost.optuna.trials"])
        else:
            from src.res.algo.boost.booster.optuna import OptunaBoostModel
            return OptunaBoostModel.DEFAULT_N_TRIALS

    @property
    def seq_lens(self) -> dict[str, int]:
        slens = dict(self["input.sequence.lens"])
        lens = {
            key: slens.get(key, slens.get(itp, 1))
            for itp, keys in self.input_keys_all.items()
            for key in keys
        }
        if self.module_type == "factor":
            lens["factor"] = 1
        return lens

    @property
    def seq_steps(self) -> dict[str, int]:
        ssteps = dict(self["input.sequence.steps"])
        steps = {
            key: ssteps.get(key, ssteps.get(itp, 1))
            for itp, keys in self.input_keys_all.items()
            for key in keys
        }
        if self.module_type == "factor":
            steps["factor"] = 1
        return steps

    @property
    def fitting_step(self) -> int:
        return int(self["train.fitting_step"])

    @property
    def train_ratio(self) -> float:
        # valid ratio is 1 - train_ratio
        return float(self["train.dataloader.train_ratio"])

    @property
    def sample_method(
        self,
    ) -> Literal["total_shuffle", "sequential", "both_shuffle", "train_shuffle"]:
        return self["train.dataloader.sample_method"]

    @property
    def shuffle_option(self) -> Literal["static", "init", "epoch"]:
        return self["train.dataloader.shuffle_option"]

    @property
    def batch_size(self) -> int:
        return int(self["train.batch_size"])

    @property
    def max_epoch(self) -> int:
        return int(self["train.max_epoch"])

    @property
    def skip_horizon(self) -> int:
        return int(self["train.skip_horizon"])

    @property
    def transfer_training(self) -> bool:
        return self["train.trainer.transfer"]

    @property
    def criterion_loss(self) -> dict[str, dict[str, Any]]:
        kwargs : dict[str, dict[str, Any]] = self["train.criterion.loss"]
        assert len(kwargs) > 0, f"{kwargs} should be not empty"
        for k, v in kwargs.items():
            if v is None:
                kwargs[k] = {}
        return {k: v for k, v in kwargs.items() if v.get('lamb' , 1.0) != 0}

    @property
    def criterion_accuracy(self) -> dict[str, dict[str, Any]]:
        kwargs = self["train.criterion.accuracy"]
        assert len(kwargs) > 0, f"{kwargs} should be not empty"
        return kwargs

    @property
    def criterion_multilosses(self) -> dict[str, dict[str, Any]]:
        kwargs = self.Param.get("train.criterion.multilosses", {})
        if kwargs:
            assert "name" in kwargs, f"{kwargs} has no name"
            assert "params" in kwargs, f"{kwargs} has no params"
            assert kwargs["name"] in ["ewa", "hybrid", "dwa", "ruw", "gls", "rws"], f"{kwargs['name']} must be one of ewa, hybrid, dwa, ruw, gls, rws"
            kwargs = {"name": kwargs["name"],"params": kwargs["params"][kwargs["name"]]}
        return kwargs

    @property
    def criterion_boost(self) -> dict[str, dict[str, Any]]:
        return self.Param.get("train.criterion.boost", {})

    @property
    def trainer_optimizer(self) -> dict[str, Any]:
        return self["train.trainer.optimizer"]

    @property
    def trainer_scheduler(self) -> dict[str, Any]:
        return self["train.trainer.scheduler"]

    @property
    def trainer_learn_rate(self) -> dict[str, Any]:
        return self["train.trainer.learn_rate"]

    @property
    def trainer_gradient_clip_value(self) -> float | None:
        return self["train.trainer.gradient.clip_value"]

    @property
    def factor_calculator(self) -> Type[FactorCalculator]:
        assert self.module_type == "factor", (
            f"{self.module_type} is not a factor module"
        )
        return StockFactorHierarchy.get_factor(self.input_factor_names[0])

    @property
    def callbackes(self) -> list[str]:
        return self["train.callbacks"]

    @property
    def callback_kwargs(self) -> dict[str, dict]:
        return {k.replace("callbacks.", ""): v for k, v in self.Param.items() if k.startswith("callbacks.")}

    @property
    def try_cuda(self) -> bool:
        return self.module_type == "nn"

    @property
    def gc_collect_each_model(self) -> bool:
        return self.module_type == "nn"

class AlgoConfig(Base.BoundLogger , Base.CacheProps):
    def __init__(
        self,
        base_path: ModelPath | Base.strPath | None,
        start_with_none: bool ,
        *,
        override: dict[str, Any] | None = None,
        module: str | None = None,
        boost_head: bool | str = False,
        short_test: bool | None = None,
        schedule_config: ScheduleConfig | None = None,
        indent: int = 1 , vb_level: Any = 2,
        **kwargs,
    ):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.base_path = ModelPath(base_path)
        self.start_with_none = start_with_none
        self.model_module = module
        self.boost_head = boost_head
        self.short_test = short_test
        self.schedule_config = schedule_config or ScheduleConfig()
        self.override = (override or {}) | kwargs
        self.load_params()
        self.override_params()
        self.check_validity()

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.model_name})"

    def load_params(self):
        if self.base_path.is_null_model:
            self.Param = get_config_dict(None)
        elif self.base_path and not self.start_with_none and (conf_file := self.base_path.conf_file(f"algo.{self.model_module}")).exists():
            self.Param = get_config_dict(conf_file)
        else:
            self.Param = get_config_dict(self.source_conf_file())
        return self

    def source_conf_file(self) -> Path:
        path = PATH.conf.joinpath("algo", self.module_type, f"{self.model_module}.yaml")
        default_path = path.with_stem(f"default")
        if not path.exists() and not default_path.exists():
            self.logger.error(f"{path} does not exist, and default.yaml does not exist either.")
        return path if path.exists() else default_path

    def override_params(self):
        self.Param.update(self.schedule_config.get(f"algo.{self.model_module}", {}))
        self.Param.update(self.override)
        return self

    def check_validity(self):
        if is_null_module_type(self.module_type):
            self.n_model = 1
        else:
            lens = [len(v) for v in self.Param.values() if isinstance(v, (list, tuple))]
            self.n_model = max(lens) if lens else 1
            if self.short_test:
                self.n_model = min(1, self.n_model)
            assert self.n_model <= 5, self.n_model

        if self.model_module == "tra":
            assert "hist_loss_seq_len" in self.Param, f"{self.Param} has no hist_loss_seq_len"
            assert "hist_loss_horizon" in self.Param, f"{self.Param} has no hist_loss_horizon"

        if self.boost_head:
            self.boost_head_config = AlgoConfig(
                self.base_path,
                start_with_none=self.start_with_none,
                module=self.boost_head,
                boost_head=False,
                schedule_config=self.schedule_config,
                override=self.override,
            )
        else:
            self.boost_head_config = None
        return self

    def expand(self):
        self.params: list[dict[str, Any]] = []

        for mm in range(self.n_model):
            par = {
                key: value[mm % len(value)] if isinstance(value, (list, tuple)) else value
                for key, value in self.Param.items()
            }
            self.params.append(par)

        if self.boost_head_config:
            self.boost_head_config.expand()
        return self

    def update_param_dict(
        self, param: dict[str, Any], key: str, value: Any, delist=True, overwrite=False
    ):
        if key in param.keys() and not overwrite:
            return
        if delist and isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if value is not None:
            param[key] = value

    def update_data_param(self, x_data: dict, config: ModelConfig):
        """when x_data is known , use it to fill some params(seq_len , input_dim , inday_dim , etc.) of nn module"""
        if not x_data:
            return self

        keys = list(x_data.keys())
        input_dim = [x_data[mdt].shape[-1] for mdt in keys]
        inday_dim = [x_data[mdt].shape[-2] for mdt in keys]
        for param in self.params:
            self.update_param_dict(param, "input_dim", input_dim)
            self.update_param_dict(param, "inday_dim", inday_dim)
            if len(keys) == 1:
                value: int = (config.seq_lens | param.get("seqlens", {})).get(keys[0], 1)
                self.update_param_dict(param, "seq_len", value)
        return self

    @property
    def base_path(self) -> ModelPath:
        return self.cached_properties.get('base_path')

    @base_path.setter
    def base_path(self, value: ModelPath):
        self.cached_properties.set('base_path' , value)

    @property
    def model_name(self) -> str:
        return self.base_path.model_name

    @property
    def module_type(self):
        return model_module_type(self.model_module)

    @property
    def model_module(self):
        if model_module := self.cached_properties.query('model_module' , lambda: None):
            return model_module
        return self.base_path.model_module
        
    @model_module.setter
    def model_module(self, value: str | None):
        self.cached_properties.set('model_module' , value)

    @property
    def boost_head(self) -> str:
        return self.cached_properties.query('boost_head' , lambda: "")

    @boost_head.setter
    def boost_head(self, value: bool | str | None):
        if not value or self.module_type != "nn":
            self.cached_properties.set('boost_head' , "")
        else:
            value = "lgbm" if value is True else value
            assert AlgoModule.is_valid(value, "boost"), f"{value} is not a valid boost module"
            self.cached_properties.set('boost_head' , value)
            
    @property
    def max_num_output(self) -> int:
        return max(self.Param.get("num_output", [1]))

@dataclass
class ModelConfigOptions:
    start: int | None = None
    end: int | None = None
    stage: int = -1
    resume: int = -1
    selection: int = -1

class ModelConfig(BaseModelConfig):
    def __init__(
        self,
        base_path: ModelPath | Base.strPath | None = None, *,
        module: str | None = None, schedule_name: str | None = None, override=None,
        start: int | None = None, end: int | None = None, stage=-1, resume=-1, selection=-1,
        indent: int = 1 , vb_level: Any = 2,
        **kwargs,
    ):
        self.set_vb(vb_level , indent)
        self.options = ModelConfigOptions(start, end, stage, resume, selection)
        self.model_config = BaseModelConfig(
            base_path, module=module, schedule_name=schedule_name, override=override, 
            indent=indent , vb_level=vb_level, **kwargs)
        self.algo_config = self.model_config.generate_algo_config()
        assert self.base_path, self.base_path
        assert self.model_config.base_path is self.base_path, \
            f"{self.model_config.base_path} != {self.base_path}"
        assert self.algo_config.base_path is self.algo_config.base_path, \
            f"{self.algo_config.base_path} != {self.algo_config.base_path}"
        
    def __repr__(self):
        return f"{self.__class__.__name__}(base_path={self.base_path})"

    @cached_property
    def device(self):
        return Device(try_cuda=self.try_cuda)

    @cached_property
    def value_dict(self) -> dict[str, Any]:
        return {}

    @property
    def schedule_config(self) -> ScheduleConfig:
        return self.model_config.schedule_config

    @property
    def boost_head_config(self) -> AlgoConfig | None:
        return self.algo_config.boost_head_config

    @classmethod
    def initialize(cls, base_path: ModelPath | Base.strPath | None, **kwargs):
        config = cls(base_path, **kwargs).start_model()
        return config

    def process_parser(self):
        """
        stage:
            [-1] , if nn / boost then choose stage, else just data + test
            [ 0] , data + fit + test
            [ 1] , data + fit
            [ 2] , data + test
        resume:
            [-1] , if not st and model_path(s) exists then choose
            [ 0] , no
            [ 1] , yes
        selection:
            [-1] , choose if optional
            [ 0] , raw model name unless fitting and not resuming
            [1,2,3,...] , choose by model_index if is_resuming
        """
        self.parser_stage(self.options.stage)
        self.parser_resume(self.options.resume)
        self.parser_select(self.options.selection)
        return self

    def start_model(self):
        self.process_parser()
        if "fit" in self.queue_of_stages and not self.is_resuming:
            if self.base_path.base.exists():
                if (not self.short_test and not self.base_path.is_null_model and self.base_path.is_resumable):
                    raise Exception(f"{self.model_name} resumable , re-train has to delete folder manually")
                self.base_path.clear_model_path()
                self.logger.alert1(f"{self.base_path} is cleared" , vb = 1)

        self.base_path.mkdir(model_nums=self.model_num_list, exist_ok=True)
        dump_kwargs = {'overwrite': self.short_test, 'vb_level': 'never'}
        self.model_config.Param.dump_yaml(self.base_path.conf_file("model") , **dump_kwargs)
        self.schedule_config.Param.dump_yaml(self.base_path.conf_file("schedule") , **dump_kwargs)
        self.algo_config.Param.dump_yaml(self.base_path.conf_file(f"algo.{self.algo_config.model_module}") , **dump_kwargs)
        if self.boost_head_config:
            self.boost_head_config.Param.dump_yaml(self.base_path.conf_file(f"algo.{self.boost_head_config.model_module}") , **dump_kwargs)
        return self

    def set_value(self, key: str, value: Any):
        self.value_dict[key] = value

    @property
    def Param(self):
        return self.model_config.Param

    @property
    def base_path(self) -> ModelPath:
        return self.model_config.base_path

    @property
    def is_null_model(self) -> bool:
        return self.base_path.is_null_model

    @property
    def full_module_name(self):
        return self.model_config.full_module_name

    @property
    def model_name(self) -> str | Any:
        return self.model_config.model_name

    @property
    def model_param(self) -> list[dict[str, Any]]:
        return self.algo_config.params

    @property
    def model_num(self) -> int:
        return self.algo_config.n_model

    @property
    def short_test(self) -> bool:
        return self.base_path.is_short_test

    @property
    def model_num_list(self) -> list[int]:
        return list(range(self.algo_config.n_model))

    @property
    def boost_head_param(self) -> dict[str, Any]:
        assert self.algo_config.boost_head_config, "boost_head_config is not set"
        assert len(self.algo_config.boost_head_config.params) == 1, self.algo_config.boost_head_config.params
        return self.algo_config.boost_head_config.params[0]

    @property
    def beg_date(self) -> int:
        beg_date = self.model_config.beg_date
        if self.module_type == "factor":
            beg_date = max(beg_date, self.factor_calculator.get_min_date())
        if self.options.start is not None:
            beg_date = max(beg_date, self.options.start)
        return beg_date

    @property
    def end_date(self) -> int:
        end_date = self.model_config.end_date
        if self.module_type == "factor":
            end_date = min(end_date, self.factor_calculator.get_max_date())
        if self.options.end is not None:
            end_date = min(end_date, self.options.end)
        return end_date

    @property
    def manual_deletion_required(self) -> bool:
        return (
            not self.short_test
            and not self.is_resuming
            and not self.base_path.is_null_model
            and "fit" in self.queue_of_stages
            and self.base_path.is_resumable
        )


    @cached_property
    def resumed_max_pred_date(self) -> int:
        """
        Resumed maximum predicted date for resumed testing. if not set, will be 19000101.
        Before this date, the predictions will not be tested again, but the model might be loaded to test dates before this date.
        
        In scenario of resuming testing from last pred date, this date is min(last predicted date, date before min(missing pred dates)).
        In scenario of resuming testing from last model date, this date is max(predicted dates of all models before last model date).
        """
        return 19000101

    @cached_property
    def queue_of_stages(self) -> list[Base.lit.StageAll]:
        """stage queue for training"""
        return []

    @cached_property
    def is_resuming(self) -> bool:
        return False

    def log_operation(self, operation: str, sub_operation: str):
        assert operation in ["fit", "test"], f"operation {operation} is not valid"
        assert sub_operation in ["start", "end"], f"sub_operation {sub_operation} is not valid"
        other_info = f"is_resuming={self.is_resuming}"
        self.base_path.log_operation(f"{operation}_model >> {sub_operation} >> {other_info}")

    def set_config_environment(self, manual_seed=None) -> None:
        self.set_random_seed(manual_seed if manual_seed else self.random_seed)
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__("allow_tf32", self["env.allow_tf32"])  # = self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self["env.detect_anomaly"])
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self, x_data: dict) -> None:
        """
        when x_data is known , use it to fill some params(seq_len , input_dim , inday_dim , etc.) of nn module
        do it in whenever x_data is changed
        """
        if self.module_type == "nn" and x_data:
            self.algo_config.update_data_param(x_data, self)

    def weight_scheme(self, stage: str, no_weight=False) -> Base.lit.ConfigWeightScheme:
        if no_weight:
            return "equal"
        stage = "fit" if stage == "fit" else "test"
        weight_scheme = str(self.model_config.Param[f"train.criterion.weight"].get(stage, "equal"))
        assert weight_scheme in ["equal" , "top" , "polar"] , weight_scheme
        return cast(Base.lit.ConfigWeightScheme, weight_scheme)

    @staticmethod
    def set_random_seed(seed=None):
        if seed is None:
            return NotImplemented
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["KMP_DUPLICATE_LIB_OK"] = ("True")  # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @property
    def parse_vb(self) -> int:
        return Proj.vb.never if self.base_path.base.exists() else 0

    def parser_stage(self, value=-1):
        """
        parser stage queue
        value:
            -1: choose if optional
            0: fit + test
            1: fit only
            2: test only
        """
        assert value in [-1, 0, 1, 2], f"stage must be -1, 0, 1, or 2, got {value}"
        queue_of_stages: list[Base.lit.StageAll] = ["data", "fit", "test"]

        if value == -1:
            if not self.base_path.is_null_model:
                msg = "--What stage would you want to run? 0: fit + test, 1: fit only , 2: test only"
                self.logger.note(msg , vb = self.parse_vb)
                value = int(input(f"[0, fit & test] , [1, fit only] , [2, test only]"))

        if self.base_path.is_null_model:
            value = 2

        match value:
            case 0:
                queue_of_stages = ["data", "fit", "test"]
            case 1:
                queue_of_stages = ["data", "fit"]
            case 2:
                queue_of_stages = ["data", "test"]
            case _:
                raise ValueError(f"Invalid stage option: {value}")

        self.queue_of_stages = queue_of_stages
        msg = "--Process Queue : {:s}".format(" + ".join(map(lambda x: (x[0].upper() + x[1:]), queue_of_stages)))
        self.logger.note(msg , vb = self.parse_vb)

    def parser_resume(self, value=-1):
        """
        parser resume flag
        value:
            -1: choose if optional
            0: no
            1: yes
        """
        assert value in [-1, 0, 1], f"resume must be -1, 0, or 1, got {value}"
        if value == -1:
            if self.short_test:
                value = 0
            else:
                candidates = self.base_path.find_resumable_candidates()
                if candidates:
                    msg = f"Multiple model path of {self.model_clean_name} exists, input [0] to deny resuming , [1] to confirm resuming!"
                    self.logger.note(msg , vb = self.parse_vb)
                    value = int(input(f"[0, not resuming] , [1, resuming]"))
                else:
                    value = 0

        match value:
            case 0:
                is_resuming = False
            case 1:
                is_resuming = True
            case _:
                raise ValueError(f"Invalid resume option: {value}")

        self.is_resuming = is_resuming
        msg = f"Confirm Resume Training!" if is_resuming else "Start Training New!"
        self.logger.note(msg , vb = self.parse_vb)

    def parser_select(self, value=-1):
        """
        parse model_name selection if model_name dirs exists
        value:

        -1: choose if optional
            if short_test or null_model or no candidates:
                don't change the base_path
            elif fit is in queue_of_stages:
                -1: choose (if model_name dirs exists, ask to choose one)
                0: raw model_name dir if is_resuming , create a new model_name dir otherwise
                1,2,3,...: choose one by model_index, start from 1 (if not is_resuming , raise error)
            elif test only:
                -1: choose (if more than 1 model_name dirs exists, ask to choose one)
                0: try to use the raw model_name dir
                1,2,3,...: choose one by model_index
        0: don't change the base_path
        """
        assert value in [-1, 0], f"initial selection must be -1 or 0, got {value}"
        candidates = self.base_path.find_resumable_candidates_indices()
        if value == -1:
            if self.short_test or self.base_path.is_null_model:
                ...
            elif "fit" in self.queue_of_stages and not self.is_resuming:
                if self.base_path.model_name_index in candidates:
                    value = self.base_path.find_new_index()
                    msg = f"ModelPath {self.base_path} is resumable, will create a new ModelPath with index {value} to Train New!"
                    self.logger.note(msg , vb = self.parse_vb)
            else:
                if len(candidates) == 1:
                    value = candidates[0]
                else:
                    msg = f"Multiple ModelPath of {self.model_clean_name} exists, input number to choose!"
                    self.logger.note(msg , vb = self.parse_vb)
                    self.logger.note(f"Options include: {candidates}" , vb = self.parse_vb)
                    value = int(input(f"Which Model to Resume?"))
                    assert value in candidates, (f"value {value} is not in candidates_indices {candidates}")
        elif value == 0:
            if self.manual_deletion_required:
                value = self.base_path.find_new_index()
            else:
                value = -1

        assert value != 0, f"value {value} is not valid"
        
        if value > 0:
            old_base = self.base_path.relative_base
            self.base_path.with_new_index(value)
            new_base = self.base_path.relative_base
            if old_base != new_base:
                msg = f'ModelPath.base {old_base} is replaced by {new_base} due to resumability!'
                self.logger.alert2(msg)

        if self.manual_deletion_required:
            msg = f"{self.base_path} resumable but choose not to resume! You have to start a new training or manually delete the existing model_name dir!"
            self.logger.error(msg)
            raise Exception(f"{self.base_path} resumable but choose not to resume!")

    def print_out(self, color: str | None = None, vb_level: Any = 2, min_key_len: int = -1):
        info_strs: list[tuple[int, str, str]] = []  # indent , key , value

        info_strs.append((0, "Module", f"{self.full_module_name}"))
        info_strs.append((0, "Model Name", self.model_name))
        if self.base_path.is_null_model:
            info_strs.append((0, "Labels", f"{self.labels}"))
            info_strs.append((0, "Period", f"{self.beg_date} ~ {self.end_date}"))
        else:
            if self.module_type == "boost":
                if self.criterion_boost.get('objective', None):
                    info_strs.append((0, "Boost Objective", f"{self.criterion_boost['objective']}"))
                if self.boost_optuna:
                    info_strs.append((0, "Boost Params", f"Optuna for {self.boost_optuna_trials} trials"))
                else:
                    info_strs.append((0, "Boost Params", ""))
                    for k, v in self.algo_config.Param.items():
                        info_strs.append((1, k, f"{v}"))
            else:
                if self.boost_head:
                    info_strs.append((0, "Use Boost Head", f"{self.boost_head}"))
                info_strs.append((0, "Model Params", ""))
                for k, v in self.algo_config.Param.items():
                    info_strs.append((1, k, f"{v}"))
            info_strs.append((0, "Model Num", f"{self.algo_config.n_model}"))
            info_strs.append((0, "Inputs", f"{self.input_type}"))
            if self.input_type == "data":
                info_strs.append((1, "Data Types", f"{self.input_data_types}"))
            elif self.input_type == "hidden":
                info_strs.append((1, "Hidden Models", f"{self.input_hidden_types}"))
            elif self.input_type == "factor":
                info_strs.append((1, "Factor Types", f"{self.input_factor_types}"))
                if self.input_factor_names:
                    info_strs.append((1, "Factor Names", f"{self.input_factor_names}"))
            elif self.input_type == "combo":
                info_strs.append((1, "Combo Types", f"{self.input_combo_types}"))
            info_strs.append((0, "Labels", f"{self.labels}"))
            info_strs.append((0, "Period", f"{self.beg_date} ~ {self.end_date}"))
            info_strs.append((0, "Interval", f"{self.interval} days"))
            info_strs.append((0, "Window", f"{self.window} days"))
            if self.module_type == "nn":
                info_strs.append((0, "Loss", f"{self.criterion_loss}"))
                info_strs.append((0, "Accuracy", f"{self.criterion_accuracy}"))
            info_strs.append((0, "Sampling", f"{self.sample_method}"))
            info_strs.append((0, "Shuffling", f"{self.shuffle_option}"))
            info_strs.append((0, "Random Seed", f"{self.random_seed}"))
        info_strs.append((0, "Short Test", f"{self.short_test}"))
        info_strs.append((0, "Stage Queue", f"{self.queue_of_stages}"))
        info_strs.append((0, "Resuming", f"{self.is_resuming}"))
        if self.is_resuming:
            info_strs.append((1, "Resume Test", f"{Const.Model.resume_test}"))
            info_strs.append((1, "Resume Test Start", f"{Const.Model.resume_test_start}"))
            info_strs.append((1, "Resume Perf", f"{Const.Model.resume_factor_perf}"))
            info_strs.append((1, "Resume FMP", f"{Const.Model.resume_fmp}"))
            info_strs.append((1, "Resume Account", f"{Const.Model.resume_fmp_account}"))

        self.logger.stdout_pairs(
            info_strs,
            title="Train Config Initiated:",
            color=color,
            vb_level=vb_level,
            min_key_len=min_key_len,
        )
        return self
