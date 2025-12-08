class Options:
    """Specified Options for the project"""
    _cache : dict[str , list[str]] = {}
    @classmethod
    def PATH(cls):
        from src.proj import PATH
        return PATH
    
    @classmethod
    def available_models(cls) -> list[str]:
        if 'available_models' not in cls._cache:
            cls._cache['available_models'] = [p.name for p in cls.PATH().model.iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]
        return cls._cache['available_models']

    @classmethod
    def available_modules(cls) -> list[str]:
        if 'available_modules' not in cls._cache:
            from src.res.algo import AlgoModule
            cls._cache['available_modules'] = [f'{module_type.replace("booster" , "boost")}/{module}' for module_type, modules in AlgoModule._availables.items() for module in modules.keys()]
        return cls._cache['available_modules']

    @classmethod
    def available_schedules(cls) -> list[str]:
        if 'available_schedules' not in cls._cache:
            cls._cache['available_schedules'] = [p.stem for p in cls.PATH().conf_schedule.glob('*.yaml')] + [p.stem for p in cls.PATH().shared_schedule.glob('*.yaml')]
        return cls._cache['available_schedules']

    @classmethod
    def available_db_mappings(cls) -> list[str]:
        if 'available_db_mappings' not in cls._cache:
            cls._cache['available_db_mappings'] = list(cls.PATH().read_yaml(cls.PATH().conf.joinpath('registry' , 'db_models_mapping')).keys())
        return cls._cache['available_db_mappings']

    @classmethod
    def available_tradeports(cls) -> list[str]:
        if 'available_tradeports' not in cls._cache:
            cls._cache['available_tradeports'] = [p.name for p in cls.PATH().trade_port.iterdir() if not p.name.startswith('.')]
        return cls._cache['available_tradeports']

    @classmethod
    def available_factors(cls) -> list[str]:
        if 'available_factors' not in cls._cache:
            from src.res.factor.calculator import FactorCalculator
            cls._cache['available_factors'] = [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling')] + \
                [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside')]
        return cls._cache['available_factors']