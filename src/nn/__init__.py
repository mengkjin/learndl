from torch import nn , Tensor
from typing import Literal
from . import (
    attention , cnn , rnn , modernTCN , patchTST , TSMixer
)

from .rnn import (simple_lstm , gru , lstm , resnet_lstm , resnet_gru ,
                  transformer , tcn , rnn_ntask , rnn_general , gru_dsize)
from .patchTST import patch_tst
from .modernTCN import modern_tcn
from .TSMixer import ts_mixer
from .tra import tra
from .factorVAE import FactorVAE as factor_vae

def get_nn_category(module_name : str) -> Literal['vae' , 'tra' , '']:
    if module_name == 'factor_vae':
        return 'vae'
    elif module_name == 'tra':
        return 'tra'
    else:
        return ''