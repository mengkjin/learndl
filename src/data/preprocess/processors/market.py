"""Market preprocessors."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.proj import DB, CALENDAR, Base
from src.data.util import DataBlock
from ..core import MicellaneousPreProcessor

class PrePro_market(MicellaneousPreProcessor):
    """
    Market-level features (key: ``'market'``).

    Combines cap-weighted daily microstructure risk (8 dims) with log returns of
    CSI300/CSI500/CSI1000 and ``microcap_400``.  Risk columns use rolling
    time-series z-score; index columns use ``log(1 + pct_chg/100)``.

    Stored with a single placeholder ``secid`` and broadcast to the stock universe
    during ``ModuleData._align_blocks``.
    """
    CalculationWindow = 120
    MIN_SAMPLES = 60
    MARKET_SECID = 0

    RISK_DB = ('market_daily', 'risk')
    INDEX_SOURCES : dict[str, tuple[str, str]] = {
        'logret_csi300': ('index_daily_ts', '000300.SH'),
        'logret_csi500': ('index_daily_ts', '000905.SH'),
        'logret_csi1000': ('index_daily_ts', '000852.SH'),
        'logret_microcap_400': ('index_daily_custom', 'microcap_400'),
    }
    RISK_FEATURES : tuple[str, ...] = (
        'market_day_true_range',
        'market_day_turnover',
        'market_day_largebuy_price_deviation',
        'market_day_smallbuy_percentage',
        'market_day_sqrt_avg_size',
        'market_day_open_close_percentage',
        'market_day_5min_ret_volatility',
        'market_day_5min_ret_skewness',
    )

    @classmethod
    def _index_date_col(cls , df : pd.DataFrame) -> str:
        if 'trade_date' in df.columns:
            return 'trade_date'
        if 'date' in df.columns:
            return 'date'
        raise ValueError(f'market index frame missing date column: {df.columns.tolist()}')

    @classmethod
    def _load_index_logret(cls , db_src : str , db_key : str) -> pd.Series:
        df = DB.load(db_src , db_key , missing_ok = True)
        if df.empty or 'pct_chg' not in df.columns:
            cls.logger.warning(f'market index {db_src}/{db_key} is empty or missing pct_chg')
            return pd.Series(dtype = float)
        date_col = cls._index_date_col(df)
        df = df.sort_values(date_col)
        logret = np.log1p(df['pct_chg'].astype(float).to_numpy() / 100.0)
        return pd.Series(logret , index = df[date_col].astype(int).to_numpy())

    @classmethod
    def _load_risk_frame(cls , start : int , end : int) -> pd.DataFrame:
        df = DB.load(cls.RISK_DB[0] , cls.RISK_DB[1] , missing_ok = True)
        if df.empty:
            cls.logger.warning(f'market risk {cls.RISK_DB} is empty')
            return pd.DataFrame()
        if 'date' not in df.columns:
            raise ValueError(f'market risk frame missing date column: {df.columns.tolist()}')
        df = df.sort_values('date').set_index('date')
        risk_cols = [c for c in cls.RISK_FEATURES if c in df.columns]
        missing = set(cls.RISK_FEATURES) - set(risk_cols)
        if missing:
            cls.logger.warning(f'market risk missing columns: {sorted(missing)}')
        if not risk_cols:
            return pd.DataFrame()
        roll = df[risk_cols].rolling(cls.CalculationWindow , min_periods = cls.MIN_SAMPLES)
        z = (df[risk_cols] - roll.mean()) / (roll.std() + 1e-6)
        z = z.loc[z.index >= start]
        return z.loc[:end] if end else z

    def pre_process(
        self , start : int | None = None , end : int | None = None , * ,
        secid : Base.alias.SecidType = None ,
        indent = 0 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs
    ) -> DataBlock:
        start = start or self.load_start
        end = end or self.load_end
        if start > end:
            return DataBlock()

        load_start = CALENDAR.td(start , -self.CalculationWindow + 1).td
        frames : list[pd.DataFrame] = []

        risk_df = self._load_risk_frame(load_start , end)
        if not risk_df.empty:
            frames.append(risk_df)

        for feat_name , (db_src , db_key) in self.INDEX_SOURCES.items():
            series = self._load_index_logret(db_src , db_key)
            if series.empty:
                continue
            frames.append(series.rename(feat_name).to_frame())

        if not frames:
            return DataBlock()

        merged = pd.concat(frames , axis = 1).sort_index()
        merged = merged.ffill().fillna(0.0).loc[start:end]
        merged = merged.dropna(how = 'all')
        if merged.empty:
            return DataBlock()

        feature_cols = merged.columns.tolist()
        out = merged.reset_index()
        if 'date' not in out.columns:
            out = out.rename(columns = {out.columns[0]: 'date'})
        out['secid'] = self.MARKET_SECID
        block = DataBlock.from_pandas(out[['secid', 'date', *feature_cols]])
        block = block.slice_date(start , end)

        secid = Base.ensure_secid(secid)
        if secid is not None and len(secid) > 1:
            values = block.values.expand(len(secid) , -1 , -1 , -1).clone()
            block = block.update(values = values , secid = secid , inplace = True)
        return block
