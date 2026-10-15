"""Labels preprocessors."""
from __future__ import annotations
import torch
from src.proj import Const
from src.func.tensor import neutralize_2d, process_factor
from src.data.util import DataBlock
from src.data.loader import BlockLoader
from ..core import TradePreProcessor

class _ReturnLabelPreProcessor(TradePreProcessor):
    """
    Shared return-label transformation; subclasses select raw database keys.

    Loads forward returns from ``labels_ts`` and CNE5 industry/size exposures.
    For each return label, adds a cross-sectionally industry+size neutralised
    version (``std*`` prefix), then applies ``process_factor`` to winsorise
    and rank all labels.
    """

    LABEL_KEYS = ('ret10_lag', 'ret20_lag', 'ret5_lag', 'ret3_lag')
    NEUTRALIZE_EXCLUDE_SUFFIXES = ('_3', '_open', '_vwap')

    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            'y' : BlockLoader('labels_ts', list(self.LABEL_KEYS), **kwargs) ,
            'risk' : BlockLoader('models', 'tushare_cne5_exp', [*Const.Factor.RISK.indus, 'size'], **kwargs)}
    def final_feat(self):
        """All features are kept (return labels + neutralised variants)."""
        return None

    def process(self , blocks : dict[str,DataBlock]):
        """Neutralise return labels and apply cross-sectional ranking/winsorising."""
        if any([block.empty for block in blocks.values()]):
            return DataBlock()
        data_block , model_exp = blocks['y'] , blocks['risk']
        indus_size = model_exp.values[...,:]
        x = torch.Tensor(indus_size).squeeze(2)
        for i_feat,lb_name in enumerate(data_block.feature):
            if lb_name.startswith('rtn') and not lb_name.endswith(self.NEUTRALIZE_EXCLUDE_SUFFIXES):
                y_raw = data_block.values[...,i_feat].squeeze(2)
                y_std = neutralize_2d(y_raw , x , dim = 0)
                assert y_std is not None , 'y_std is None'
                y_std = y_std.unsqueeze(2)
                data_block.add_feature('std'+lb_name[3:],y_std)

        y_ts = data_block.values[:,:,0]
        for i_feat,lb_name in enumerate(data_block.feature):
            y_pro = process_factor(y_ts[...,i_feat], dim = 0)
            if y_pro is None: 
                continue
            data_block.values[...,i_feat] = y_pro.unsqueeze(-1)

        return data_block


class PrePro_y(_ReturnLabelPreProcessor):
    """Default lag1 return labels; preserve existing fields and transforms."""


class PrePro_y_lag0(_ReturnLabelPreProcessor):
    """Independent 5/10/20-day lag0 return and neutralised labels."""
    LABEL_KEYS = ("ret10", "ret20", "ret5")


class PrePro_y_open(_ReturnLabelPreProcessor):
    """5/10/20-day open-to-open returns, always from T+1 to T+1+n."""
    LABEL_KEYS = ('ret10_open', 'ret20_open', 'ret5_open')
    NEUTRALIZE_EXCLUDE_SUFFIXES = ()


class PrePro_y_vwap(_ReturnLabelPreProcessor):
    """5/10/20-day VWAP-to-VWAP returns, always from T+1 to T+1+n."""
    LABEL_KEYS = ('ret10_vwap', 'ret20_vwap', 'ret5_vwap')
    NEUTRALIZE_EXCLUDE_SUFFIXES = ()
