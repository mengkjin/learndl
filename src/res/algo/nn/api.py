from torch import nn
from typing import Literal

from . import layer as Layer
from . import model as Model
from .util import get_multiloss_params

AVAILABLE_NNS = {
    'simple_lstm'       : Model.Recurrent.simple_lstm,
    'gru'               : Model.Recurrent.gru, 
    'lstm'              : Model.Recurrent.lstm, 
    'resnet_lstm'       : Model.Recurrent.resnet_lstm, 
    'resnet_gru'        : Model.Recurrent.resnet_gru,
    'transformer'       : Model.Recurrent.transformer, 
    'tcn'               : Model.Recurrent.tcn, 
    'rnn_ntask'         : Model.Recurrent.rnn_ntask, 
    'rnn_general'       : Model.Recurrent.rnn_general, 
    'gru_dsize'         : Model.Recurrent.gru_dsize, 
    'patch_tst'         : Model.PatchTST.patch_tst, 
    'modern_tcn'        : Model.ModernTCN.modern_tcn, 
    'ts_mixer'          : Model.TSMixer.ts_mixer, 
    'tra'               : Model.TRA.tra, 
    'factor_vae'        : Model.FactorVAE.FactorVAE,
    'risk_att_gru'      : Model.RiskAttGRU.risk_att_gru,
    'ple_gru'           : Model.PLE.ple_gru,
    'tft'               : Model.TFT.TemporalFusionTransformer
}

def get_nn_module(module_name : str) -> nn.Module:
    return AVAILABLE_NNS[module_name]

def get_nn_category(module_name : str) -> str | None:
    default_category = getattr(AVAILABLE_NNS.get(module_name , None) , '_default_category' , None)
    if default_category:
        return default_category
    elif module_name == 'factor_vae':
        return 'vae'
    elif module_name == 'tra':
        return 'tra' 
    else:
        return None
    
def get_nn_datatype(module_name : str) -> str | None:
    default_data_type = getattr(AVAILABLE_NNS.get(module_name , None) , '_default_data_type' , None)
    if default_data_type:
        return default_data_type
    elif module_name == 'risk_att_gru':
        return 'day+style+indus'
    else:
        return None
    

