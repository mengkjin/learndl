"""Trade preprocessors."""
from __future__ import annotations
import numpy as np
from src.proj import CALENDAR
from src.data.util import DataBlock
from src.data.loader import BlockLoader
from ..core import TradePreProcessor
from .registry import TRADE_ALLOW_INACTIVE

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

class PrePro_15m(TradePreProcessor):
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
    15-minute bar preprocessor (key: ``'15m'``).

    Prices are divided by the daily pre-close. Volume is converted to free-float
    turnover by allocating day ``turn_fl`` according to each bar's share of the
    day's intraday volume sum (avoids day/minute volume unit mismatch), then
    clipped at 200%. The ``volume`` feature is renamed to ``turn_fl``.
    """
    AllowInactive : bool = TRADE_ALLOW_INACTIVE
    def block_loaders(self , **kwargs) -> dict[str,BlockLoader]:
        return {
            '15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap'], **kwargs)}

    def process(self , blocks):
        data_block = blocks['15m']
        return data_block
    
class PrePro_30m(TradePreProcessor):
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

class PrePro_30mcont(TradePreProcessor):
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
    
class PrePro_60m(TradePreProcessor):
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
    
class PrePro_week(TradePreProcessor):
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
    
