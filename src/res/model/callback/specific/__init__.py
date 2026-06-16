from typing import Type

from src.proj import Logger
from src.res.model.util import ModelConfig, BaseCallBack

def get_specific_cbs(config : ModelConfig) -> list[type[BaseCallBack]]:
    
    special = config.special
    nn_category = config.module_type
    module_name = config.model_module
    model_name = config.model_clean_name

    cbs : list[type[BaseCallBack]] = []
    if special.get('dsize') or module_name == 'gru_dsize':
        from src.res.model.callback.specific.dsize import SpecificCB_DSize
        cbs.append(SpecificCB_DSize)
    if special.get('vae') or nn_category == 'vae':
        from src.res.model.callback.specific.vae import SpecificCB_VAE
        cbs.append(SpecificCB_VAE)
    if special.get('tra') or nn_category == 'tra':
        from src.res.model.callback.specific.tra import SpecificCB_TRA
        cbs.append(SpecificCB_TRA)
    if special.get('global2top') or module_name.endswith('_global2top') or model_name.endswith('_global2top'):
        from src.res.model.callback.specific.global2top import SpecificCB_Global2Top
        cbs.append(SpecificCB_Global2Top)

    return cbs