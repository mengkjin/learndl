import json , sys

from datetime import datetime
from src.proj.env import MACHINE , PATH

class OptionsDefinition:
    """Specified Options for the project , how they are defined/calculated"""
    @classmethod
    def available_models(cls) -> list[str]:
        """Get the available nn/booster models in the models directory"""
        return [p.name for p in PATH.model.iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]

    @classmethod
    def available_modules(cls) -> list[str]:
        """Get the available nn/booster modules in the src/res/algo directory"""
        sys.stderr.write(f'Redefine available modules at {datetime.now()}\n')
        from src.res.algo import AlgoModule
        return [f'{module_type.replace("booster" , "boost")}/{module}' for module_type, modules in AlgoModule._availables.items() for module in modules.keys()]

    @classmethod
    def available_schedules(cls) -> list[str]:
        """Get the available schedules in the config/schedule directory"""
        return [p.stem for p in PATH.conf.joinpath('schedule').glob('*.yaml')] + [p.stem for p in PATH.shared_schedule.glob('*.yaml')]

    @classmethod
    def available_db_mappings(cls) -> list[str]:
        """Get the available db mappings in the config/proj/model_settings.yaml file"""
        return list(MACHINE.configs('proj' , 'model_settings')['db_mapping'].keys())

    @classmethod
    def available_tradeports(cls) -> list[str]:
        """Get the available trade ports in the trade_port directory"""
        return [p.name for p in PATH.trade_port.iterdir() if not p.name.startswith('.')]

    @classmethod
    def available_factors(cls) -> list[str]:
        """Get the available factors in the of pooling and sellside categories"""
        sys.stderr.write(f'Redefine available factors at {datetime.now()}\n')
        from src.res.factor.calculator import FactorCalculator
        return [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling' , updatable = True)] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside' , updatable = True)]

class OptionsCache:
    """Cache for the options , used to accelerate the streamlit interactive app"""
    cache_path = PATH.local_machine.joinpath('options_cache.json')
    cache : dict[str , list[str]] = {}

    def __init__(self):
        if not self.cache_path.exists():
            json.dump({}, self.cache_path.open('w'))
        self.cache = json.load(self.cache_path.open('r'))

    def get(self , key : str) -> list[str]:
        """Get the options from the cache"""
        if key not in self.cache:
            self.cache[key] = getattr(OptionsDefinition , key)()
            json.dump(self.cache, self.cache_path.open('w'))
        return self.cache[key]

    @classmethod
    def update(cls):
        """update the cache stored in cache_path locally"""
        cache = {}
        for method in dir(OptionsDefinition):
            if not method.startswith(('_')):
                cache[method] = getattr(OptionsDefinition , method)()
        json.dump(cache, cls.cache_path.open('w'))

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
    def available_models(cls) -> list[str]:
        """Get the available nn/booster models from the cache"""
        return cls.cache.get('available_models')

    @classmethod
    def available_modules(cls) -> list[str]:
        """Get the available nn/booster modules from the cache"""
        return cls.cache.get('available_modules')

    @classmethod
    def available_schedules(cls) -> list[str]:
        """Get the available schedules from the cache"""
        return cls.cache.get('available_schedules')

    @classmethod
    def available_db_mappings(cls) -> list[str]:
        """Get the available db mappings from the cache"""
        return cls.cache.get('available_db_mappings')

    @classmethod
    def available_tradeports(cls) -> list[str]:
        """Get the available trade ports from the cache"""
        return cls.cache.get('available_tradeports')

    @classmethod
    def available_factors(cls) -> list[str]:
        """Get the available factors from the cache"""
        return cls.cache.get('available_factors')