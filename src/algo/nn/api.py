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
    'tft'               : Model.TemporalFusionTransformer.TemporalFusionTransformer
}

def valid_nn(nn_type : str):
    return nn_type in AVAILABLE_NNS

def get_nn_module(module_name : str) -> nn.Module:
    return AVAILABLE_NNS[module_name]

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
    

