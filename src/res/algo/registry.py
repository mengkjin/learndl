"""Model names and constructor paths; safe to import when building CLI options."""
from __future__ import annotations

__all__ = [
    'NN_MODULES',
    'FACTOR_SEQUENCE_MODULES',
    'BOOST_MODULES',
]

NN_MODULES = {
    'simple_lstm': 'RNN.simple_lstm',
    'gru': 'RNN.gru',
    'lstm': 'RNN.lstm',
    'transformer': 'RNN.transformer',
    'tcn': 'RNN.tcn',
    'rnn_ntask': 'RNN.rnn_ntask',
    'rnn_general': 'RNN.rnn_general',
    'gru_dsize': 'RNN.gru_dsize',
    'patch_tst': 'PatchTST.patch_tst',
    'modern_tcn': 'ModernTCN.modern_tcn',
    'ts_mixer': 'TSMixer.ts_mixer',
    'tra': 'TRA.tra',
    'factor_vae': 'FactorVAE.FactorVAE',
    'risk_att_gru': 'RiskAttGRU.risk_att_gru',
    'ple_gru': 'PLE.ple_gru',
    'moe_gru': 'MoE.moe_gru',
    'tft': 'TFT.TemporalFusionTransformer',
    'astgnn': 'gnn.Astgnn',
    'astgnn_legacy': 'gnn.AstgnnLegacy',
    'resnet_gru': 'resnet_gru.resnet_gru',
    'gru_gru': 'resnet_gru.gru_gru',
    'gru_gru_new': 'resnet_gru.gru_gru_new',
    'transformer_gru': 'transformer_gru.transformer_gru',
    'resnet_transformer': 'resnet_transformer.resnet_transformer',
    'resnet_mlp': 'resnet_mlp.resnet_mlp',
}

FACTOR_SEQUENCE_MODULES = {
    'factorseq_mlp': 'FactorSnapshot',
    'factorseq_gru': 'FactorGRU',
    'factorseq_m1': 'FactorLateFusion',
    'factorseq_m2': 'FactorConditioned',
    'factorseq_mask': 'FactorMissingness',
    'factorseq_gate': 'FactorValueGate',
}

BOOST_MODULES = {
    'ada': 'ada.AdaBoost',
    'lgbm': 'lgbm.Lgbm',
    'xgboost': 'xgboost.XgBoost',
    'catboost': 'catboost.CatBoost',
}

