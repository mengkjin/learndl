from typing import Type

from src.res.algo import AlgoModule
from src.res.model.util import BaseCallBack

def get_specific_cb(module_name : str) -> Type[BaseCallBack] | None:
    nn_category = AlgoModule.nn_category(module_name)
    if module_name == 'gru_dsize':
        from src.res.model.callback.specific.dsize import SpecificCB_DSize
        return SpecificCB_DSize
    elif nn_category == 'vae':
        from src.res.model.callback.specific.vae import SpecificCB_VAE
        return SpecificCB_VAE
    elif nn_category == 'tra':
        from src.res.model.callback.specific.tra import SpecificCB_TRA
        return SpecificCB_TRA
    elif module_name.endswith('_global2top'):
        from src.res.model.callback.specific.global2top import SpecificCB_Global2Top
        return SpecificCB_Global2Top
    else:
        return None