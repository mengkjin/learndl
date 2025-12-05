class Options:
    """Specified Options for the project"""
    @classmethod
    def PATH(cls):
        from src.proj import PATH
        return PATH
    
    @classmethod
    def available_models(cls) -> list[str]:
        return [p.name for p in cls.PATH().model.iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]

    @classmethod
    def available_modules(cls) -> list[str]:
        from src.res.algo import AlgoModule
        return [f'{module_type.replace("booster" , "boost")}/{module}' for module_type, modules in AlgoModule._availables.items() for module in modules.keys()]

    @classmethod
    def available_schedules(cls) -> list[str]:
        return [p.stem for p in cls.PATH().conf_schedule.glob('*.yaml')] + [p.stem for p in cls.PATH().shared_schedule.glob('*.yaml')]

    @classmethod
    def available_db_mappings(cls) -> list[str]:
        return list(cls.PATH().read_yaml(cls.PATH().conf.joinpath('registry' , 'db_models_mapping')).keys())

    @classmethod
    def available_tradeports(cls) -> list[str]:
        return [p.name for p in cls.PATH().trade_port.iterdir() if not p.name.startswith('.')]

    @classmethod
    def available_factors(cls) -> list[str]:
        from src.res.factor.calculator import FactorCalculator
        return [p.factor_name for p in FactorCalculator.iter(meta_type = 'pooling')] + \
            [p.factor_name for p in FactorCalculator.iter(category1 = 'sellside')]