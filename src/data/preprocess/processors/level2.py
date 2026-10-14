"""Level2 preprocessors."""
from __future__ import annotations
import polars as pl
from src.proj import DB, CALENDAR, Base
from src.data.util import DataBlock
from ..core import MicellaneousPreProcessor

class PrePro_dfl2(MicellaneousPreProcessor):
    """
    Dongfang L2 rolling percentage rank (key: ``'dfl2'``).

    Source ``sellside/dongfang.l2_chars`` is frozen.  Code is kept so existing
    dumps can still be loaded; batch preprocess skips this key (``ENABLED=False``).
    """
    ENABLED = False
    CalculationWindow = 250
    MIN_SAMPLES = 90
    FEATURE_CHUNK_SIZE = 20
    def pre_process(
        self , start : int | None = None , end : int | None = None , * , 
        secid : Base.alias.SecidType = None , 
        indent = 0 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs) -> DataBlock:
        """
        Load Dongfang L2 chars and apply per-secid rolling z-score normalisation.

        Features are processed in chunks of ``FEATURE_CHUNK_SIZE`` to limit peak
        memory.  The rolling window is ``CalculationWindow`` bars with a minimum
        of ``MIN_SAMPLES`` effective observations.

        Note: the Polars expression currently has a parenthesisation bug —
        ``.alias()`` is applied to the std denominator before division, and
        ``+1e-6`` is added to the numerator instead of the denominator.
        See TODO_data.md item C4 for the fix.
        """
        # 1. load data into pl.DataFrame
        start = start or self.load_start
        df = DB.loads_pl('sellside', 'dongfang.l2_chars', start = CALENDAR.td(start , -self.CalculationWindow + 1).td , end = end , key_column = None , vb_level = vb_level)
        if len(df) == 0:
            return DataBlock()
        secid = Base.ensure_secid(secid)
        if secid is not None:
            df = df.filter(pl.col('secid').is_in(secid))
        # 2. Identify the columns as features (exclude index columns)
        feature = [c for c in df.columns if c not in ['secid', 'date']]

        # 3. Apply rolling pct_rank partitioned over secid
        df = df.sort(['secid', 'date'])
        blocks = []
        for i in range(0, len(feature), self.FEATURE_CHUNK_SIZE):
            sub_feature = feature[i:i + self.FEATURE_CHUNK_SIZE]
            # sub_df = df.select(['secid', 'date'] + sub_feature).lazy().with_columns([
            #     ((pl.col(feat) - pl.col(feat).rolling_mean(window_size=self.CalculationWindow, min_samples=self.MIN_SAMPLES).over("secid")) /
            #     (pl.col(feat).rolling_std(window_size=self.CalculationWindow, min_samples=self.MIN_SAMPLES).over("secid") + 1e-6)).alias(feat)
            #     for feat in sub_feature
            # ]).collect()
            sub_df = (
                df.lazy().select(['secid', 'date'] + sub_feature).sort(['secid', 'date'])
                .with_columns(
                    pl.lit(1).alias("_one")
                )
                .with_columns(
                    pl.col(feat).rolling_rank(window_size=self.CalculationWindow, min_samples=self.MIN_SAMPLES).over("secid").alias(f"{feat}_rank")
                    for feat in sub_feature
                )
                .with_columns(
                    pl.col("_one").rolling_sum(window_size=self.CalculationWindow).over("secid").alias("window_n")
                )
                .with_columns(
                    (pl.col(f"{feat}_rank") / pl.col("window_n")).alias(feat) for feat in sub_feature
                )
                .drop(["_one", "window_n"] + [f"{feat}_rank" for feat in sub_feature])
                .collect()
            )
            blocks.append(DataBlock.from_polars(sub_df).slice_date(start , end))
        del df
        return DataBlock.merge(blocks , inplace = True)

class PrePro_dfl2cs(MicellaneousPreProcessor):
    """
    Dongfang L2 characteristics with cross-sectional z-score (key: ``'dfl2cs'``).

    Source ``sellside/dongfang.l2_chars`` is frozen.  Code is kept so existing
    dumps can still be loaded; batch preprocess skips this key (``ENABLED=False``).

    Note: same parenthesisation bug as ``PrePro_dfl2`` — see TODO_data.md item C4.
    """
    ENABLED = False
    FEATURE_CHUNK_SIZE = 20

    def pre_process(
        self , start : int | None = None , end : int | None = None , * , 
        secid : Base.alias.SecidType = None , 
        indent = 0 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs
    ) -> DataBlock:
        """
        Load Dongfang L2 chars and apply cross-sectional z-score normalisation.

        Features are processed in chunks of ``FEATURE_CHUNK_SIZE`` to limit peak memory.
        """
        # 1. load data into pl.DataFrame
        start = start or self.load_start
        df = DB.loads_pl('sellside', 'dongfang.l2_chars', start = CALENDAR.td(start , -self.CalculationWindow + 1).td , end = end , key_column = None , vb_level = vb_level)
        if len(df) == 0:
            return DataBlock()
        secid = Base.ensure_secid(secid)
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

