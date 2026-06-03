from .config import *

def check_all_config_files():     
    from src.res.model.util.config.inspector import ModelConfigsInspector
    from src.res.model.util.config.modifier import ModelConfigsBatchModifier
    from src.proj import Logger
    Logger.stdout('Checking all config files...')
    modifier = ModelConfigsBatchModifier()
    modifier.batch_modify()
    inspecter = ModelConfigsInspector()
    inspecter.inspect_key_values()
    Logger.success('All config files checked.')
