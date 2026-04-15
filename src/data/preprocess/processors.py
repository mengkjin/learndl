"""
All concrete ``PrePro_*`` preprocessor classes and the ``PrePros`` registry accessor.

Each ``PrePro_<key>`` class is auto-registered in ``PreProcessorMeta.registry`` on
import.  ``PrePros`` is a thin facade over the registry for use by ``ModuleData``
and ``PreProcessorTask``.

Registered preprocessors
-------------------------
- ``y``              : Return labels + risk-neutralised variants
- ``day``            : Daily adjusted OHLCV (price-adjusted)
- ``15m`` / ``30m`` / ``60m`` : Intraday bars normalised by daily preclose and turnover
- ``week``           : 5-day rolling OHLCV reshaped into the inday dimension
- ``style``          : CNE5 style factor exposures
- ``indus``          : CNE5 industry factor exposures
- ``quality`` / ``growth`` / ``value`` / ``earning`` / ``surprise`` / ``coverage`` /
  ``forecast`` / ``adjustment`` / ``hf_*`` / ``momentum`` / ``volatility`` /
  ``correlation`` / ``liquidity`` / ``holding`` / ``trading`` : Factor category preprocessors
- ``dfl2``           : Dongfang L2 characteristics (rolling time-series z-score)
- ``dfl2cs``         : Dongfang L2 characteristics (cross-sectional z-score)
"""
from __future__ import annotations

import torch
import numpy as np
import polars as pl

from typing import Any , Literal

from src.proj import Proj , DB , CALENDAR , CONST
from src.func.tensor import neutralize_2d , process_factor
from src.data.util import DataBlock
from src.data.loader import BlockLoader

from .core import PreProcessor , FactorPreProcessor , TradePreProcessor , MicellaneousPreProcessor

class PrePros:
    """
    Thin registry facade over ``PreProcessorMeta.registry``.

    Provides key listing, processor instantiation, and iteration helpers
    used by ``ModuleData`` and ``PreProcessorTask``.
    """
    @classmethod
    def keys(cls) -> list[str]:
        """Return all registered preprocessor keys."""
        return [name for name in PreProcessor.registry.keys()]

    @classmethod
    def start_date(cls , type : Literal['fit' , 'predict'] = 'predict') -> int:
        """Return the start date used for the given ``type`` across all processors."""
        return PreProcessor.start_date(type)

    @classmethod
    def get_processor(cls , key : str , type : Literal['fit' , 'predict'] , **kwargs) -> PreProcessor:
        """Instantiate and return the preprocessor registered under ``key``."""
        return PreProcessor.registry[key](type , **kwargs)

    @classmethod
    def iter_processors(cls , type : Literal['fit' , 'predict'] , **kwargs):
        """Iterate over all registered processors, yielding instantiated objects."""
        for key in PreProcessor.registry.keys():
            yield cls.get_processor(key , type)

class PrePro_y(TradePreProcessor):
    """
    Return label preprocessor (key: ``'y'``).

    Loads forward return labels (10-day and 20-day, with 1-day lag) from
    ``labels_ts`` and CNE5 industry/size exposures from the risk model.
    For each return label, adds a cross-sectionally industry+size neutralised
    version (``std*`` prefix), then applies ``process_factor`` to winsorise
    and rank all labels.
    """
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'y' : BlockLoader('labels_ts', ['ret10_lag', 'ret20_lag']) ,
                'risk' : BlockLoader('models', 'tushare_cne5_exp', [*CONST.Conf.Factor.RISK.indus, 'size'])}
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
            if lb_name.startswith('rtn'):
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

class PrePro_day(TradePreProcessor):
    """Daily adjusted OHLCV preprocessor (key: ``'day'``).  Applies adjfactor to price columns."""
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'day' : BlockLoader('trade_ts', 'day', ['adjfactor', *self.final_feat()])}

    def process(self , blocks):
        """Apply price adjustment and return the result."""
        return blocks['day'].adjust_price()

class PrePro_15m(TradePreProcessor):
    """
    15-minute bar preprocessor (key: ``'15m'``).

    Prices are divided by the daily pre-close; volume is scaled by daily
    free-float turnover / daily volume to produce a turnover fraction.
    The ``volume`` feature is renamed to ``turn_fl``.
    """
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'15m' : BlockLoader('trade_ts', '15min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}

    def process(self , blocks):
        data_block = blocks['15m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_30m(TradePreProcessor):
    """30-minute bar preprocessor (key: ``'30m'``).  Same normalisation as ``PrePro_15m``."""
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'30m' : BlockLoader('trade_ts', '30min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}

    def process(self , blocks): 
        data_block = blocks['30m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_60m(TradePreProcessor):
    """60-minute bar preprocessor (key: ``'60m'``).  Same normalisation as ``PrePro_15m``."""
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'60m' : BlockLoader('trade_ts', '60min', ['close', 'high', 'low', 'open', 'volume', 'vwap']) ,
                'day' : BlockLoader('trade_ts', 'day', ['volume', 'turn_fl', 'preclose'])}

    def process(self , blocks):
        data_block = blocks['60m']
        db_day     = blocks['day'].align(data_block.secid , data_block.date , inplace = True)
        
        data_block = data_block.adjust_price(divide = db_day.loc(feature = 'preclose'))
        data_block = data_block.adjust_volume(multiply = db_day.loc(feature = 'turn_fl') , 
                                              divide = db_day.loc(feature = 'volume') + 1e-6, 
                                              vol_feat = 'volume')
        data_block = data_block.rename_feature({'volume':'turn_fl'})
        return data_block
    
class PrePro_week(TradePreProcessor):
    """
    5-day rolling OHLCV preprocessor (key: ``'week'``).

    Reshapes daily bars into a (N_secid, N_date, 5, N_feature) tensor where
    the inday dimension holds the trailing 5 days.  Prices are normalised by
    the Monday (inday=0) pre-close so the window is stationary across weeks.
    """
    WEEKDAYS = 5

    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'day':BlockLoader('trade_ts', 'day', ['adjfactor', 'preclose', *self.final_feat()])}
    
    def load_blocks(self , start = None , end = None , secid = None , indent = 0 , vb_level : Any = 1 , **kwargs):
        vb_level = Proj.vb(vb_level)
        if start is not None and start < 0: 
            start = 2 * start
        elif start is not None and start > 0:
            start = CALENDAR.td(start , -self.WEEKDAYS + 1).td
        blocks : dict[str,DataBlock] = {}
        date = CALENDAR.range(start , end)
        for src_key , loader in self.block_loaders().items():
            blocks[src_key] = loader.load(start , end , indent = indent + 1 , vb_level = vb_level + 1 , **kwargs).align_secid_date(secid , date , inplace = True)
            secid = blocks[src_key].secid
        return blocks
    
    def process(self , blocks): 
        data_block = blocks['day'].adjust_price()

        new_values = np.full(np.multiply(data_block.shape,(1, 1, self.WEEKDAYS, 1)),np.nan)
        for i in range(self.WEEKDAYS): 
            new_values[:,self.WEEKDAYS-1-i:,i] = data_block.values[:,:len(data_block.date)-self.WEEKDAYS+1+i,0]
        data_block.update(values = new_values)
        data_block = data_block.adjust_price(adjfactor = False , divide=data_block.loc(inday = 0,feature = 'preclose'))
        return data_block
    
class PrePro_style(PreProcessor):
    """CNE5 style factor exposures (key: ``'style'``)."""
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'style' : BlockLoader('models', 'tushare_cne5_exp', CONST.Conf.Factor.RISK.style)}

    def final_feat(self):
        """Keep all style features."""
        return None

    def process(self , blocks):
        """Return the raw style exposure block unchanged."""
        return blocks['style']

class PrePro_indus(PreProcessor):
    """CNE5 industry factor exposures (key: ``'indus'``)."""
    def block_loaders(self) -> dict[str,BlockLoader]:
        return {'indus' : BlockLoader('models', 'tushare_cne5_exp', CONST.Conf.Factor.RISK.indus)}

    def final_feat(self):
        """Keep all industry features."""
        return None

    def process(self , blocks):
        """Return the raw industry exposure block unchanged."""
        return blocks['indus']

class PrePro_quality(FactorPreProcessor):
    """Quality factor preprocessor (key: ``'quality'``)."""

class PrePro_growth(FactorPreProcessor):
    """Growth factor preprocessor (key: ``'growth'``)."""

class PrePro_value(FactorPreProcessor):
    """Value factor preprocessor (key: ``'value'``)."""

class PrePro_earning(FactorPreProcessor):
    """Earnings factor preprocessor (key: ``'earning'``)."""

class PrePro_surprise(FactorPreProcessor):
    """Earnings surprise factor preprocessor (key: ``'surprise'``)."""

class PrePro_coverage(FactorPreProcessor):
    """Analyst coverage factor preprocessor (key: ``'coverage'``)."""

class PrePro_forecast(FactorPreProcessor):
    """Earnings forecast factor preprocessor (key: ``'forecast'``)."""

class PrePro_adjustment(FactorPreProcessor):
    """Analyst revision/adjustment factor preprocessor (key: ``'adjustment'``)."""

class PrePro_hf_momentum(FactorPreProcessor):
    """High-frequency momentum factor preprocessor (key: ``'hf_momentum'``)."""

class PrePro_hf_volatility(FactorPreProcessor):
    """High-frequency volatility factor preprocessor (key: ``'hf_volatility'``)."""

class PrePro_hf_correlation(FactorPreProcessor):
    """High-frequency correlation factor preprocessor (key: ``'hf_correlation'``)."""

class PrePro_hf_liquidity(FactorPreProcessor):
    """High-frequency liquidity factor preprocessor (key: ``'hf_liquidity'``)."""

class PrePro_momentum(FactorPreProcessor):
    """Price momentum factor preprocessor (key: ``'momentum'``)."""

class PrePro_volatility(FactorPreProcessor):
    """Price volatility factor preprocessor (key: ``'volatility'``)."""

class PrePro_correlation(FactorPreProcessor):
    """Return correlation factor preprocessor (key: ``'correlation'``)."""

class PrePro_liquidity(FactorPreProcessor):
    """Liquidity factor preprocessor (key: ``'liquidity'``)."""

class PrePro_holding(FactorPreProcessor):
    """Institutional holding factor preprocessor (key: ``'holding'``)."""

class PrePro_trading(FactorPreProcessor):
    """Trading behaviour factor preprocessor (key: ``'trading'``)."""

class PrePro_dfl2(MicellaneousPreProcessor):
    """
    Calculate the rolling z-score of the features partitioned over secid
    """
    CALCULATION_WINDOW = 250
    MIN_SAMPLES = 90
    FEATURE_CHUNK_SIZE = 20
    def pre_process(self , start : int | None = None , end : int | None = None , * , secid : np.ndarray | None = None , indent = 0 , vb_level : Any = 'max' , **kwargs) -> DataBlock:
        """
        Load Dongfang L2 chars and apply per-secid rolling z-score normalisation.

        Features are processed in chunks of ``FEATURE_CHUNK_SIZE`` to limit peak
        memory.  The rolling window is ``CALCULATION_WINDOW`` bars with a minimum
        of ``MIN_SAMPLES`` valid observations.

        Note: the Polars expression currently has a parenthesisation bug —
        ``.alias()`` is applied to the std denominator before division, and
        ``+1e-6`` is added to the numerator instead of the denominator.
        See TODO_data.md item C4 for the fix.
        """
        # 1. load data into pl.DataFrame
        start = start or self.load_start
        df = DB.loads_pl('sellside', 'dongfang.l2_chars', start = CALENDAR.td(start , -self.CALCULATION_WINDOW + 1).td , end = end , key_column = None , vb_level = vb_level)
        if len(df) == 0:
            return DataBlock()
        if secid is not None:
            df = df.filter(pl.col('secid').is_in(secid))
        # 2. Identify the columns as features (exclude index columns)
        feature = [c for c in df.columns if c not in ['secid', 'date']]

        # 3. Apply rolling z-score partitioned over secid
        df = df.sort(['secid', 'date'])
        blocks = []
        for i in range(0, len(feature), self.FEATURE_CHUNK_SIZE):
            sub_feature = feature[i:i + self.FEATURE_CHUNK_SIZE]
            sub_df = df.select(['secid', 'date'] + sub_feature).lazy().with_columns([
                ((pl.col(feat) - pl.col(feat).rolling_mean(window_size=self.CALCULATION_WINDOW, min_samples=self.MIN_SAMPLES).over("secid")) /
                (pl.col(feat).rolling_std(window_size=self.CALCULATION_WINDOW, min_samples=self.MIN_SAMPLES).over("secid") + 1e-6)).alias(feat)
                for feat in sub_feature
            ]).collect()
            blocks.append(DataBlock.from_polars(sub_df).slice_date(start , end))
        del df
        return DataBlock.merge(blocks , inplace = True)

class PrePro_dfl2cs(MicellaneousPreProcessor):
    """
    Dongfang L2 characteristics with cross-sectional z-score (key: ``'dfl2cs'``).

    For each date, subtracts the cross-sectional mean and divides by the
    cross-sectional std.  Features are processed in chunks of ``FEATURE_CHUNK_SIZE``.

    Note: same parenthesisation bug as ``PrePro_dfl2`` — see TODO_data.md item C4.
    """
    FEATURE_CHUNK_SIZE = 20

    def pre_process(self , start : int | None = None , end : int | None = None , * , secid : np.ndarray | None = None , indent = 0 , vb_level : Any = 'max' , **kwargs) -> DataBlock:
        """
        Load Dongfang L2 chars and apply cross-sectional z-score normalisation.

        Features are processed in chunks of ``FEATURE_CHUNK_SIZE`` to limit peak memory.
        """
        # 1. load data into pl.DataFrame
        start = start or self.load_start
        df = DB.loads_pl('sellside', 'dongfang.l2_chars', start = CALENDAR.td(start , -self.CALCULATION_WINDOW + 1).td , end = end , key_column = None , vb_level = vb_level)
        if len(df) == 0:
            return DataBlock()
        if secid is not None:
            df = df.filter(pl.col('secid').is_in(secid))
        # 2. Identify the columns as features (exclude index columns)
        feature = [c for c in df.columns if c not in ['secid', 'date']]

        # 3. Apply rolling z-score partitioned over secid
        df = df.sort(['secid', 'date'])
        blocks = []
        for i in range(0, len(feature), self.FEATURE_CHUNK_SIZE):
            sub_feature = feature[i:i + self.FEATURE_CHUNK_SIZE]
            sub_df = df.select(['secid', 'date'] + sub_feature).lazy().with_columns([
                ((pl.col(feat) - pl.col(feat).mean().over("date")) / (pl.col(feat).std().over("date") + 1e-6)).alias(feat) for feat in sub_feature
            ]).collect()
            blocks.append(DataBlock.from_polars(sub_df).slice_date(start , end))
        del df
        return DataBlock.merge(blocks , inplace = True)
