"""Trade preprocessors."""
from __future__ import annotations
import numpy as np
import torch
from src.proj import CALENDAR
from src.data.util import DataBlock
from src.data.loader import BlockLoader
from ..core import TradePreProcessor
from .registry import TRADE_ALLOW_INACTIVE

class _VwapInfCloseMixin:
    """Repair only infinite VWAP, before transforms and after the final cast."""
    def _repair_vwap_inf(self, block, stage):
        if block.empty or not {'vwap', 'close'}.issubset(block.feature):
            return
        i_vwap, i_close = [list(block.feature).index(f) for f in ('vwap', 'close')]
        observer = getattr(self, 'vwap_inf_repair_observer', None)
        for start in range(0, len(block.date), 64):
            values = block.values[:, start:start + 64]
            vwap, close = values[..., i_vwap], values[..., i_close]
            mask = torch.isinf(vwap)
            if not mask.any():
                continue
            finite_close = torch.isfinite(close)
            if observer is not None:
                observer({'stage': stage, 'start': int(block.date[start]),
                          'end': int(block.date[min(start + 63, len(block.date) - 1)]),
                          'inf_count': int(mask.sum()),
                          'close_nonfinite': int((mask & ~finite_close).sum())})
            # Invalid close stays missing; never copy an Inf into VWAP.
            vwap[mask] = torch.where(finite_close[mask], close[mask], torch.nan)

    def process_blocks(self, blocks):
        for source, block in blocks.items():
            self._repair_vwap_inf(block, f'source:{source}')
        block = super().process_blocks(blocks)
        self._repair_vwap_inf(block, 'processed_float32')
        return block

class PrePro_day(TradePreProcessor):
    """Daily adjusted OHLCV preprocessor (key: ``'day'``).  Applies adjfactor to price columns."""
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        final_feat = self.final_feat() or []
        day = BlockLoader('trade_ts', 'day', ['adjfactor', *final_feat], **kwargs)
        return {'day':day}

    def process(self , blocks):
        """Apply price adjustment and return the result."""
        return blocks['day'].adjust_price()

class PrePro_15m(_VwapInfCloseMixin, TradePreProcessor):
    """
    15-minute bar preprocessor (key: ``'15m'``).

    Prices are divided by the daily pre-close. Volume is converted to free-float
    turnover by allocating day ``turn_fl`` according to each bar's share of the
    day's intraday volume sum (avoids day/minute volume unit mismatch), then
    clipped at 200%. The ``volume`` feature is renamed to ``turn_fl``.
    """
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            '15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap'], **kwargs) ,
            'day' : BlockLoader('trade_ts', 'day', ['turn_fl', 'preclose'], **kwargs)}

    def process(self , blocks):
        data_block = blocks['15m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        return self.allocate_turn_fl_by_bar_volume(data_block , db_day.loc(feature = 'turn_fl'))

class PrePro_15mraw(TradePreProcessor):
    """
    Raw 15-minute OHLCV bars (key: ``'15mraw'``).

    Keep raw prices and volume; window normalisation happens at model input.
    """
    AllowInactive : bool = TRADE_ALLOW_INACTIVE

    def final_feat(self) -> list[str]:
        # The parent expects turn_fl, which would discard volume and insert NaNs.
        return ['open', 'close', 'high', 'low', 'vwap', 'volume']

    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            '15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap'], **kwargs)}

    def process(self , blocks):
        data_block = blocks['15m']
        return data_block

    def process_blocks(self, blocks):
        # Run after the base float32 cast so cast-induced Inf is covered too.
        block = super().process_blocks(blocks)
        if block.empty:
            return block
        observer = getattr(self, 'vwap_inf_observer', None)
        if observer is not None:
            observer(block)  # Autofix records the evidence before replacement.
        i_vwap, i_close, i_volume = [list(block.feature).index(f) for f in ('vwap', 'close', 'volume')]
        for start in range(0, len(block.date), 64):
            values = block.values[:, start:start + 64]
            vwap, close = values[..., i_vwap], values[..., i_close]
            fallback = torch.isinf(vwap) | (values[..., i_volume] == 0)
            values[..., i_vwap] = torch.where(
                fallback, torch.where(torch.isfinite(close), close, torch.nan), vwap)
        return block
    
class PrePro_30m(_VwapInfCloseMixin, TradePreProcessor):
    """30-minute bar preprocessor (key: ``'30m'``).  Same normalisation as ``PrePro_15m``."""
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            '30m' : BlockLoader('trade_ts', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap'], **kwargs) ,
            'day' : BlockLoader('trade_ts', 'day', ['turn_fl', 'preclose'], **kwargs)}

    def process(self , blocks): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        return self.allocate_turn_fl_by_bar_volume(data_block , db_day.loc(feature = 'turn_fl'))

class PrePro_30mcont(_VwapInfCloseMixin, TradePreProcessor):
    """30-minute bar preprocessor (key: ``'30mcont'``).  Only convert volume to turn_fl."""
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            '30m' : BlockLoader('trade_ts', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap'], **kwargs) ,
            'day' : BlockLoader('trade_ts', 'day', ['turn_fl'], **kwargs)}

    def process(self , blocks): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        return self.allocate_turn_fl_by_bar_volume(data_block , db_day.loc(feature = 'turn_fl'))
    
class PrePro_60m(_VwapInfCloseMixin, TradePreProcessor):
    """60-minute bar preprocessor (key: ``'60m'``).  Same normalisation as ``PrePro_15m``."""
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            '60m' : BlockLoader('trade_ts', '60min', ['close', 'high', 'low', 'open', 'volume', 'vwap'], **kwargs) ,
            'day' : BlockLoader('trade_ts', 'day', ['turn_fl', 'preclose'], **kwargs)}

    def process(self , blocks):
        data_block = blocks['60m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        return self.allocate_turn_fl_by_bar_volume(data_block , db_day.loc(feature = 'turn_fl'))
    
class PrePro_week(_VwapInfCloseMixin, TradePreProcessor):
    """
    5-day rolling OHLCV preprocessor (key: ``'week'``).

    Reshapes daily bars into a (N_secid, N_date, 5, N_feature) tensor where
    the inday dimension holds the trailing 5 days.  Prices are normalised by
    the Monday (inday=0) pre-close so the window is stationary across weeks.
    """
    Weekdays = 5
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        final_feat = self.final_feat() or []
        day = BlockLoader('trade_ts', 'day', ['adjfactor', 'preclose', *final_feat], **kwargs)
        return {'day':day}
    
    def load_blocks(self , start = None , end = None , secid = None , **kwargs):
        if start is not None and start < 0: 
            start = 2 * start
        elif start is not None and start > 0:
            start = CALENDAR.td(start , -self.Weekdays + 1).td
        blocks : dict[str,DataBlock] = {}
        date = CALENDAR.range(start , end)
        block_loaders = self.block_loaders(indent = self.indent + 1 , vb_level = self.vb_level + 3)
        for src_key , loader in block_loaders.items():
            blocks[src_key] = loader.load(start , end , **kwargs).align_secid_date(secid , date , inplace = True)
            secid = blocks[src_key].secid
        return blocks
    
    def process(self , blocks): 
        data_block = blocks['day'].adjust_price(ffill = True).adjust_volume(ffill = True)

        new_values = np.full(np.multiply(data_block.shape,(1, 1, self.Weekdays, 1)),np.nan)
        for i in range(self.Weekdays): 
            new_values[:,self.Weekdays-1-i:,i] = data_block.values[:,:len(data_block.date)-self.Weekdays+1+i,0]
        new_block = DataBlock(values = new_values , secid = data_block.secid , date = data_block.date , feature = data_block.feature)
        new_block = new_block.adjust_price(adjfactor = False , divide=new_block.loc(inday = 0,feature = 'preclose'))
        return new_block
    
