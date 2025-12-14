import json

from datetime import datetime
from .path import PATH

class OptionsDefinition:
    """Specified Options for the project"""
    @classmethod
    def available_models(cls) -> list[str]:
        return [p.name for p in PATH.model.iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]

    @classmethod
    def available_modules(cls) -> list[str]:
        print(f'refind available modules at {datetime.now()}')
        from src.res.algo import AlgoModule
        return [f'{module_type.replace("booster" , "boost")}/{module}' for module_type, modules in AlgoModule._availables.items() for module in modules.keys()]

    @classmethod
    def available_schedules(cls) -> list[str]:
        return [p.stem for p in PATH.conf_schedule.glob('*.yaml')] + [p.stem for p in PATH.shared_schedule.glob('*.yaml')]

    @classmethod
    def available_db_mappings(cls) -> list[str]:
        return list(PATH.read_yaml(PATH.conf.joinpath('registry' , 'db_models_mapping')).keys())

    @classmethod
    def available_tradeports(cls) -> list[str]:
        return [p.name for p in PATH.trade_port.iterdir() if not p.name.startswith('.')]

    @classmethod
    def available_factors(cls) -> list[str]:
        print(f'refind available factors at {datetime.now()}')
        from src.res.factor.calculator import FactorCalculator
        return [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling' , updatable = True)] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside' , updatable = True)]

class OptionsCache:
    """Cache for the options"""
    cache_path = PATH.local_machine.joinpath('options_cache.json')
    cache : dict[str , list[str]] = {}

    def __init__(self):
        if not self.cache_path.exists():
            json.dump({}, self.cache_path.open('w'))
        self.cache = json.load(self.cache_path.open('r'))

    def get(self , key : str) -> list[str]:
        if key not in self.cache:
            self.cache[key] = getattr(OptionsDefinition , key)()
            json.dump(self.cache, self.cache_path.open('w'))
        return self.cache[key]

    @classmethod
    def update(cls):
        cache = {}
        for method in dir(OptionsDefinition):
            if not method.startswith(('_')):
                cache[method] = getattr(OptionsDefinition , method)()
        json.dump(cache, cls.cache_path.open('w'))

    @classmethod
    def clear(cls):
        cls.cache_path.unlink()
        cls.cache = {}

class Options:
    """Specified Options for the project"""
    cache = OptionsCache()

    @classmethod
    def update(cls):
        cls.cache.update()
    
    @classmethod
    def available_models(cls) -> list[str]:
        return cls.cache.get('available_models')

    @classmethod
    def available_modules(cls) -> list[str]:
        return cls.cache.get('available_modules')

    @classmethod
    def available_schedules(cls) -> list[str]:
        return cls.cache.get('available_schedules')

    @classmethod
    def available_db_mappings(cls) -> list[str]:
        return cls.cache.get('available_db_mappings')

    @classmethod
    def available_tradeports(cls) -> list[str]:
        return cls.cache.get('available_tradeports')

    @classmethod
    def available_factors(cls) -> list[str]:
        return cls.cache.get('available_factors')