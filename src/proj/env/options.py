"""Discover models, modules, schedules, ports, and factors; cache lists for Streamlit and CLI."""
from __future__ import annotations
from datetime import datetime
from src.proj.env import PATH

__all__ = ['OptionsDefinition' , 'OptionsCache' , 'Options']

class OptionsDefinition:
    """Specified Options for the project , how they are defined/calculated"""
    @classmethod
    def available_models(cls) -> list[str]:
        """Get the available nn/boost models in the models directory"""
        models : list[str] = []
        for root in [PATH.model_nn , PATH.model_boost , PATH.model_st]:
            for model in root.iterdir():
                if model.is_dir() and not model.name.startswith('.'):
                    models.append(f'{root.name}/{model.name}')
        return models

    @classmethod
    def available_modules(cls) -> list[str]:
        """Get the available nn/boost modules in the src/res/algo directory"""
        from src.res.algo import AlgoModule
        from src.proj.log import Logger
        Logger.info(f'Redefine available modules at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        return [f'{module_type}/{module}' for module_type, modules in AlgoModule._availables.items() for module in modules.keys()]

    @classmethod
    def available_schedules(cls) -> list[str]:
        """Get the available schedules in the config/model/schedule directory"""
        return sorted([p.stem for p in PATH.sched.glob('*.yaml')] + [p.stem for p in PATH.sched_shared.glob('*.yaml')])

    @classmethod
    def available_trackingports(cls) -> list[str]:
        """Get the available tracking ports in the trade_port directory"""
        return [p.name for p in PATH.trade_port.iterdir() if not p.name.startswith('.')]

    @classmethod
    def available_backtestports(cls) -> list[str]:
        """Get the available backtest ports in the trade_port directory"""
        return [p.name for p in PATH.rslt_trade.joinpath('backtest').iterdir() if not p.name.startswith('.')]

    @classmethod
    def available_factors(cls) -> list[str]:
        """Get the available factors in the of pooling and sellside categories"""
        from src.res.factor.calculator import FactorCalculator
        from src.proj.log import Logger
        Logger.info(f'Redefine available factors at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        return [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling' , updatable = True)] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside' , updatable = True)]

class OptionsCache:
    """Cache for the options , used to accelerate the streamlit interactive app"""
    cache_path = PATH.cache.joinpath('options_cache.json')
    cache : dict[str , list[str]] = {}

    def __init__(self):
        """Load JSON cache from disk; create empty file if missing."""
        if not self.cache_path.exists():
            PATH.dump_json({}, self.cache_path)

    def ensure_load_cache(self):
        if not hasattr(self, '_loaded'):
            self.cache.update(PATH.read_json(self.cache_path))
            self._loaded = True

    def get(self , key : str , refresh : bool = False) -> list[str]:
        """Get the options from the cache"""
        self.ensure_load_cache()
        if key not in self.cache or refresh:
            self.cache[key] = getattr(OptionsDefinition , key)()
            PATH.dump_json(self.cache, self.cache_path , overwrite = True)
        return self.cache[key]

    @classmethod
    def update(cls):
        """update the cache stored in cache_path locally"""
        cls.clear()
        for method in dir(OptionsDefinition):
            if not method.startswith('_'):
                cls.cache[method] = getattr(OptionsDefinition , method)()
        PATH.dump_json(cls.cache, cls.cache_path , overwrite = True)

    @classmethod
    def clear(cls):
        """clear the cache and cache_path"""
        cls.cache_path.unlink(missing_ok=True)
        cls.cache = {}

class Options:
    """Specified Options for the project"""
    cache = OptionsCache()

    @classmethod
    def update(cls):
        """update the cached options"""
        cls.cache.update()
    
    @classmethod
    def available_models(cls , refresh : bool = False) -> list[str]:
        """Get the available nn/boost models from the cache"""
        return cls.cache.get('available_models' , refresh)

    @classmethod
    def available_modules(cls , refresh : bool = False) -> list[str]:
        """Get the available nn/boost modules from the cache"""
        return cls.cache.get('available_modules' , refresh)

    @classmethod
    def available_schedules(cls , refresh : bool = False) -> list[str]:
        """Get the available schedules from the cache"""
        return cls.cache.get('available_schedules' , refresh)

    @classmethod
    def available_trackingports(cls , refresh : bool = False) -> list[str]:
        """Get the available trade ports from the cache"""
        return cls.cache.get('available_trackingports' , refresh)

    @classmethod
    def available_backtestports(cls , refresh : bool = False) -> list[str]:
        """Get the available backtest ports from the cache"""
        return cls.cache.get('available_backtestports' , refresh)

    @classmethod
    def available_factors(cls , refresh : bool = False) -> list[str]:
        """Get the available factors from the cache"""
        return cls.cache.get('available_factors' , refresh)