from torch import nn
from typing import Literal
from . import (
    Recurrent, Attention , CNN , ModernTCN , layer , PatchTST , TSMixer , TRA , FactorVAE ,
    RiskAttGRU , PLE
)
from .util import get_multiloss_params

AVAILABLE_MODULES = {
    'simple_lstm'       : Recurrent.simple_lstm,
    'gru'               : Recurrent.gru, 
    'lstm'              : Recurrent.lstm, 
    'resnet_lstm'       : Recurrent.resnet_lstm, 
    'resnet_gru'        : Recurrent.resnet_gru,
    'transformer'       : Recurrent.transformer, 
    'tcn'               : Recurrent.tcn, 
    'rnn_ntask'         : Recurrent.rnn_ntask, 
    'rnn_general'       : Recurrent.rnn_general, 
    'gru_dsize'         : Recurrent.gru_dsize, 
    'patch_tst'         : PatchTST.patch_tst, 
    'modern_tcn'        : ModernTCN.modern_tcn, 
    'ts_mixer'          : TSMixer.ts_mixer, 
    'tra'               : TRA.tra, 
    'factor_vae'        : FactorVAE.FactorVAE,
    'risk_att_gru'      : RiskAttGRU.risk_att_gru,
    'ple_gru'           : PLE.ple_gru
}

def get_nn_module(module_name : str) -> nn.Module:
    return AVAILABLE_MODULES[module_name]

def get_nn_category(module_name : str) -> Literal['vae' , 'tra' , '']:
    if module_name == 'factor_vae':
        return 'vae'
    elif module_name == 'tra':
        return 'tra'
    else:
        return ''
    
def get_nn_datatype(module_name : str) -> str | None:
    if module_name == 'risk_att_gru':
        return 'day+style+indus'
    else:
        return None
    

