"""Risk preprocessors."""
from __future__ import annotations
from src.proj import Const
from src.data.loader import BlockLoader
from ..core import PreProcessor

class PrePro_style(PreProcessor):
    """CNE5 style factor exposures (key: ``'style'``)."""
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            'style' : BlockLoader('models', 'tushare_cne5_exp', Const.Factor.RISK.style, **kwargs)}

    def final_feat(self):
        """Keep all style features."""
        return None

    def process(self , blocks):
        """Return the raw style exposure block unchanged."""
        return blocks['style']

class PrePro_indus(PreProcessor):
    """CNE5 industry factor exposures (key: ``'indus'``)."""
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            'indus' : BlockLoader('models', 'tushare_cne5_exp', Const.Factor.RISK.indus, **kwargs)}

    def final_feat(self):
        """Keep all industry features."""
        return None

    def process(self , blocks):
        """Return the raw industry exposure block unchanged."""
        return blocks['indus']

