"""NN architecture registry for the algo module.

``AVAILABLE_NNS`` maps config string keys to model constructors.  Use
``get_nn_module(name)`` to retrieve a constructor and instantiate it.

Special class attributes read by the training loop:
    _default_category:  ``'tra'`` signals TRA training loop (requires
                        ``hist_loss`` and ``y`` as extra forward args);
                        ``'vae'`` signals VAE training loop (requires ``y``).
    _default_data_type: ``'day+style+indus'`` signals that the model expects
                        a 3-tuple input (trade, style, indus) instead of a
                        plain daily feature tensor.
"""
from torch import nn

from . import layer as Layer
from . import model as Model

__all__ = [
    'AVAILABLE_NNS' , 'get_nn_module' , 'get_nn_category' , 'get_nn_datatype' ,
    'Layer' , 'Model']

# Mapping from registry key to model constructor.
# 18 registered architectures as of 2026-04.
AVAILABLE_NNS = {
    'simple_lstm'       : Model.RNN.simple_lstm,
    'gru'               : Model.RNN.gru, 
    'lstm'              : Model.RNN.lstm, 
    'resnet_lstm'       : Model.RNN.resnet_lstm, 
    'resnet_gru'        : Model.RNN.resnet_gru,
    'transformer'       : Model.RNN.transformer, 
    'tcn'               : Model.RNN.tcn, 
    'rnn_ntask'         : Model.RNN.rnn_ntask, 
    'rnn_general'       : Model.RNN.rnn_general, 
    'gru_dsize'         : Model.RNN.gru_dsize, 
    'patch_tst'         : Model.PatchTST.patch_tst, 
    'modern_tcn'        : Model.ModernTCN.modern_tcn, 
    'ts_mixer'          : Model.TSMixer.ts_mixer, 
    'tra'               : Model.TRA.tra, 
    'factor_vae'        : Model.FactorVAE.FactorVAE,
    'risk_att_gru'      : Model.RiskAttGRU.risk_att_gru,
    'ple_gru'           : Model.PLE.ple_gru,
    'tft'               : Model.TFT.TemporalFusionTransformer,
    'astgnn'            : Model.ABCM.Astgnn,
}

def get_nn_module(module_name : str) -> nn.Module:
    """Return the model constructor for a given registry key.

    Args:
        module_name: Registry key string (e.g. ``'gru'``, ``'patch_tst'``).

    Returns:
        The model class (not an instance).  Call it with the appropriate
        ``(input_dim, hidden_dim, ...)`` arguments to create a model.

    Raises:
        KeyError: If ``module_name`` is not in ``AVAILABLE_NNS``.
    """
    return AVAILABLE_NNS[module_name]

def get_nn_category(module_name : str) -> str | None:
    """Return the special training-loop category for a model, or ``None``.

    Reads ``_default_category`` from the model class.  Falls back to hard-coded
    exceptions for models registered before the class attribute was introduced.

    Returns:
        ``'tra'``  — TRA training loop required
        ``'vae'``  — VAE training loop required
        ``None``   — standard training loop
    """
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
    """Return the required data variant for a model, or ``None``.

    Reads ``_default_data_type`` from the model class.  Falls back to hard-coded
    exceptions.

    Returns:
        ``'day+style+indus'`` — training data must include style and industry
                                risk factors alongside the daily features
        ``None``              — standard daily features only
    """
    default_data_type = getattr(AVAILABLE_NNS.get(module_name , None) , '_default_data_type' , None)
    if default_data_type:
        return default_data_type
    elif module_name == 'risk_att_gru':
        return 'day+style+indus'
    else:
        return None
    

